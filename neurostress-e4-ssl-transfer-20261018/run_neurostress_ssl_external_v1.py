#!/usr/bin/env python3
"""Frozen label-free external E4 SSL transfer evaluation for NeuroStress HCI."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cognifyguard.ssl_only import MODALITIES, normalize_records
from cognifyguard.structured_faults import FAMILIES, LEVELS, structured_fault
from run_emowear_ssl_external_v1 import (CUTOFF, MECH_ROOT, MECH_SELECTION, V7_ROOT,
    V7_SELECTION, digest, model_specs, scores_for, verify_models)
from train_ssl_only_pilot import collate

BASE = ROOT / "reports/imwut_2026/neurostress_ssl_external_v1"
DATA = ROOT / "data/raw_external/neuro-stress-resilience-hci-1.0.0/selected_downloads"
PREFLIGHT = BASE / "signal_only_preflight_v1.json"
OUTPUT = BASE / "external_evaluation_v1"
RATES = {"acc": 32, "bvp": 64, "eda": 4, "temp": 4}
CHANNELS = {"ACC": "acc", "BVP": "bvp", "EDA": "eda", "TEMP": "temp"}
OUTCOMES = BASE / "participant_model_scores.csv"
DEPOSIT_RECEIPT = BASE / "evaluator_deposit_receipt.json"
PARALLELISM_AMENDMENT = BASE / "parallelism_amendment_deposit_receipt.json"
PARALLEL_WORKERS = 6
_WORKER_RECORDS = None


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def load_records():
    preflight = json.loads(PREFLIGHT.read_text())
    if preflight["status"] != "SIGNAL_ONLY_PREFLIGHT_PASS_NO_MODEL_SCORING" or preflight["model_scores_computed"]:
        raise ValueError("Signal-only preflight status is not a pre-outcome pass")
    checksums = {}
    for line in (DATA / "metadata_SHA256SUMS.txt").read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            checksums[fields[1].strip()] = fields[0].lower()
    records = {}
    participants = [f"p{i:02d}" for i in range(1, 38) if i not in (14, 23)]
    for pid in participants:
        streams, starts, sizes = {}, [], []
        for channel, modality in CHANNELS.items():
            path = DATA / f"{pid}_{channel}.csv"
            if not path.is_file() or sha(path) != checksums.get(f"PPG/{pid}/{channel}.csv"):
                raise ValueError(f"Publisher checksum mismatch: {pid}/{channel}")
            with path.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                start_row, rate_row = next(reader), next(reader)
                start = np.asarray([float(v) for v in start_row], dtype=np.float64)
                rate = np.asarray([float(v) for v in rate_row], dtype=np.float64)
                expected_width = 3 if modality == "acc" else 1
                if len(start) != expected_width or len(rate) != expected_width:
                    raise ValueError(f"Unexpected E4 header shape: {pid}/{channel}")
                if not np.isfinite(start).all() or not np.isfinite(rate).all():
                    raise ValueError(f"Nonfinite E4 header: {pid}/{channel}")
                if np.max(start) - np.min(start) > 1e-9 or np.max(rate) - np.min(rate) > 1e-9:
                    raise ValueError(f"ACC axis header disagreement: {pid}/{channel}")
                if abs(rate[0] - RATES[modality]) > 1e-9:
                    raise ValueError(f"Unexpected native rate: {pid}/{channel}")
                starts.append(float(start[0]))
                array = np.asarray([[float(v) for v in row] for row in reader if row], dtype=np.float64)
            if array.ndim == 1:
                array = array[:, None]
            if array.ndim != 2 or array.shape[1] != expected_width or not np.isfinite(array).all():
                raise ValueError(f"Nonfinite or malformed signal values: {pid}/{channel}")
            streams[modality] = array.T.astype(np.float32)
            sizes.append(array.shape[0])
        if max(starts) - min(starts) > .125:
            raise ValueError(f"Start alignment gate failed: {pid}")
        nwin = min(n // (10 * RATES[m]) for n, m in zip(sizes, MODALITIES, strict=True))
        if nwin < 100:
            raise ValueError(f"Minimum eligible windows failed: {pid}")
        # The E4 native ACC CSV is raw counts; frozen WESAD convention is counts/15.625 (= g).
        streams["acc"] = streams["acc"] / 15.625
        token = hashlib.sha256(f"neurostress_ssl_external_v1|{pid.lower()}".encode()).hexdigest()[:16]
        windows = []
        for idx in range(nwin):
            modalities = {m: streams[m][:, idx * 10 * RATES[m]:(idx + 1) * 10 * RATES[m]].copy()
                          for m in MODALITIES}
            windows.append({"participant_id": token, "window_index": idx, "modalities": modalities})
        records[token] = windows
    if len(records) != 35:
        raise ValueError("Expected all 35 publisher-eligible participants")
    return records


def evaluation_cases(source, norm):
    clean = source["modalities"]
    token, idx = source["participant_id"], source["window_index"]
    source_norm = normalize_records([source], norm)[0]["modalities"]
    for family in FAMILIES:
        for level in LEVELS:
            for modality in MODALITIES:
                seed_text = f"structured_faults_v3|{token}|{idx}|{family}|{modality}"
                seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
                faulted, dose = structured_fault(clean[modality], modality, family, level,
                    seed=seed, training_mean=norm[modality]["mean"])
                if not dose["nonzero_dose"]:
                    continue
                values = {m: clean[m].copy() for m in MODALITIES}
                values[modality] = faulted
                yield modality, {"participant_id": token, "window_index": idx, "modalities": values}, source_norm


def infer(spec, records):
    norm = spec["payload"]["normalization"]
    risk_sum, retained, total = defaultdict(float), defaultdict(int), defaultdict(int)
    clean_rejected, clean_total = defaultdict(int), defaultdict(int)
    for token, windows in records.items():
        for start in range(0, len(windows), 32):
            batch_cases = []
            clean_batch = windows[start:start+32]
            clean_x = collate(normalize_records(clean_batch, norm))["modalities"]
            with torch.inference_mode():
                clean_ev = scores_for(spec, clean_x)
            for score_name, clean_score in clean_ev.items():
                clean_rejected[(token, score_name)] += int((clean_score.max(axis=1) > CUTOFF).sum())
                clean_total[(token, score_name)] += len(clean_batch)
            for source in clean_batch:
                for modality, case, source_norm in evaluation_cases(source, norm):
                    batch_cases.append((modality, case, source_norm))
            for offset in range(0, len(batch_cases), 128):
                group = batch_cases[offset:offset+128]
                normalized = normalize_records([x[1] for x in group], norm)
                x = collate(normalized)["modalities"]
                with torch.inference_mode():
                    ev_by_score = scores_for(spec, x)
                observed = {m: x[m].cpu().numpy() for m in MODALITIES}
                losses = np.mean([np.mean((observed[m] - np.stack([g[2][m] for g in group])) ** 2,
                                          axis=(1, 2)) for m in MODALITIES], axis=0)
                for score_name, ev in ev_by_score.items():
                    rejected = ev.max(axis=1) > CUTOFF
                    for i, (modality, _, _) in enumerate(group):
                        total[(score_name, token, modality)] += 1
                        if not rejected[i]:
                            retained[(score_name, token, modality)] += 1
                            risk_sum[(score_name, token, modality)] += float(losses[i])
    rows = []
    for token in records:
        for score_name in spec["scores"]:
            risks, coverages = [], []
            for modality in MODALITIES:
                n, keep = total[(score_name, token, modality)], retained[(score_name, token, modality)]
                rows.append({"participant_token": token, "model": spec["job"]["fold"] + f"_seed{spec['job']['seed']}",
                    "score": score_name, "modality": modality, "fault_total": n, "retained": keep,
                    "coverage": keep / n if n else float("nan"),
                    "retained_standardized_mse": risk_sum[(score_name, token, modality)] / keep if keep else float("nan")})
                risks.append(risk_sum[(score_name, token, modality)] / keep if keep else float("nan"))
                coverages.append(keep / n if n else float("nan"))
            rows.append({"participant_token": token, "model": spec["job"]["fold"] + f"_seed{spec['job']['seed']}",
                "score": score_name, "modality": "equal_modality_mean", "fault_total": sum(total[(score_name,token,m)] for m in MODALITIES),
                "retained": sum(retained[(score_name,token,m)] for m in MODALITIES), "coverage": float(np.mean(coverages)),
                "retained_standardized_mse": float(np.mean(risks)) if np.isfinite(risks).all() else float("nan"),
                "clean_windows": len(records[token]),
                "clean_alert_rate": clean_rejected[(token,score_name)] / clean_total[(token,score_name)] if clean_total[(token,score_name)] else float("nan")})
    return rows


def analyze(rows):
    """Participant-paired inference with the protocol's fixed zero-coverage rule."""
    models = sorted({r["model"] for r in rows})
    participants = sorted({r["participant_token"] for r in rows})
    scores = ("physiome_raw_1nn", "physiome_restoration", "derivative_roughness_hybrid")
    cells = {(r["participant_token"], r["model"], r["score"], r["modality"]): r for r in rows}
    participant_values = {}
    zero = defaultdict(list)
    for token in participants:
        for score in scores:
            model_risks, model_coverages = [], []
            for model in models:
                modality_risks, modality_coverages = [], []
                for modality in MODALITIES:
                    r = cells[(token, model, score, modality)]
                    n, keep = int(r["fault_total"]), int(r["retained"])
                    if n <= 0 or keep <= 0:
                        zero[(token, score)].append(f"{model}:{modality}")
                        modality_risks.append(float("nan"))
                        modality_coverages.append(float("nan"))
                    else:
                        modality_risks.append(float(r["retained_standardized_mse"]))
                        modality_coverages.append(keep / n)
                model_risks.append(float(np.mean(modality_risks)) if np.isfinite(modality_risks).all() else float("nan"))
                model_coverages.append(float(np.mean(modality_coverages)) if np.isfinite(modality_coverages).all() else float("nan"))
            participant_values[(token, score)] = (float(np.mean(model_risks)) if np.isfinite(model_risks).all() else float("nan"),
                float(np.mean(model_coverages)) if np.isfinite(model_coverages).all() else float("nan"))

    comparisons, pvals = [], []
    for j, ssl in enumerate(scores[:2]):
        risk_diff, cov_diff = [], []
        for token in participants:
            sr, sc = participant_values[(token, ssl)]
            hr, hc = participant_values[(token, "derivative_roughness_hybrid")]
            risk_diff.append(sr - hr)
            cov_diff.append(sc - hc)
        risk_diff, cov_diff = np.asarray(risk_diff), np.asarray(cov_diff)
        estimable = bool(np.isfinite(risk_diff).all() and np.isfinite(cov_diff).all())
        if estimable:
            rng = np.random.default_rng(20261017)
            boots = np.asarray([risk_diff[rng.integers(0, len(risk_diff), len(risk_diff))].mean() for _ in range(10000)])
            cov_boots = np.asarray([cov_diff[rng.integers(0, len(cov_diff), len(cov_diff))].mean() for _ in range(10000)])
            rngp = np.random.default_rng(20261018)
            signs = rngp.choice(np.array([-1., 1.]), size=(100000, len(risk_diff)))
            p = float((1 + np.sum(np.abs((signs * risk_diff).mean(axis=1)) >= abs(risk_diff.mean()))) / 100001)
            cov_lower = float(np.quantile(cov_boots, .05))
            risk_ci = [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]
        else:
            p, cov_lower, risk_ci = 1.0, None, None
        pvals.append(p)
        comparisons.append({"contrast": f"{ssl} minus derivative_roughness_hybrid", "n_participants": len(participants),
            "estimable": estimable,
            "zero_retained_participant_model_modality_cells": {t: zero[(t, ssl)] for t in participants if zero[(t, ssl)]},
            "mean_paired_risk_difference": float(risk_diff.mean()) if estimable else None,
            "risk_bootstrap_95_percentile_ci": risk_ci, "paired_sign_flip_two_sided_p": p if estimable else None,
            "coverage_difference_mean": float(cov_diff.mean()) if estimable else None,
            "coverage_difference_one_sided_95_percentile_lower_bound": cov_lower,
            "coverage_noninferiority_pass": bool(cov_lower is not None and cov_lower > -.05)})
    order = np.argsort(pvals)
    adjusted = [None, None]
    running = 0.
    for rank, idx in enumerate(order):
        running = max(running, min(1., (len(pvals) - rank) * pvals[idx]))
        adjusted[idx] = running
    for row, adj in zip(comparisons, adjusted, strict=True):
        row["holm_adjusted_p"] = float(adj)
        row["risk_superiority_pass"] = bool(row["estimable"] and row["mean_paired_risk_difference"] < 0 and adj < .05)
        row["joint_success"] = bool(row["risk_superiority_pass"] and row["coverage_noninferiority_pass"])
    return {"protocol_id": "neurostress_ssl_external_v1", "inference_unit": "participant",
        "participant_count": len(participants), "frozen_model_count": len(models),
        "annotation_or_task_prediction": False, "primary_contrasts": comparisons,
        "any_ssl_joint_success": any(r["joint_success"] for r in comparisons),
        "claim_scope": "Frozen SSL rejection on deterministic synthetic E4 faults only; no natural signal quality or task/stress claims."}


def _initialize_worker(records):
    global _WORKER_RECORDS
    _WORKER_RECORDS = records
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _infer_worker(spec):
    return infer(spec, _WORKER_RECORDS)


def main():
    if OUTCOMES.exists():
        raise FileExistsError("Preserve prior evaluation attempts; refusing overwrite")
    preflight = json.loads(PREFLIGHT.read_text())
    if preflight["annotations_or_non_e4_signals_read"] or preflight["target_models_retrained_or_refit"]:
        raise ValueError("Preflight scope/provenance mismatch")
    v7, baseline = verify_models()
    specs = model_specs(v7, baseline)
    if len(specs) != 18:
        raise ValueError("Expected nine frozen SSL and nine hybrid models")
    # Repeatability check is signal/seed-only; it computes no scores.
    records = load_records()
    s = next(iter(records.values()))[0]
    a = list(evaluation_cases(s, specs[0]["payload"]["normalization"]))
    b = list(evaluation_cases(s, specs[0]["payload"]["normalization"]))
    if len(a) != len(b) or any(not np.array_equal(x[1]["modalities"][m], y[1]["modalities"][m])
        for x, y in zip(a, b, strict=True) for m in MODALITIES):
        raise ValueError("Fault generation is not deterministic")
    source_paths = ["scripts/run_neurostress_ssl_external_v1.py", "scripts/run_emowear_ssl_external_v1.py",
        "scripts/train_ssl_mechanism_development.py", "scripts/train_ssl_only_pilot.py",
        "src/cognifyguard/structured_faults.py", "src/cognifyguard/ssl_only.py",
        "src/cognifyguard/ssl_mechanism.py", "src/cognifyguard/physiome_native.py",
        "configs/neurostress_ssl_external_v1.json",
        "configs/neurostress_ssl_external_v1_format_amendment.json",
        "configs/neurostress_ssl_external_v1_identity_amendment.json",
        "configs/neurostress_ssl_external_v1_parallelism_amendment.json",
        "reports/imwut_2026/neurostress_ssl_external_v1/signal_only_preflight_v1.json",
        "reports/imwut_2026/neurostress_ssl_external_v1/partial_attempt_20261018.json",
        "reports/imwut_2026/neurostress_ssl_external_v1/parallelism_amendment_deposit_receipt.json",
        "reports/imwut_2026/ssl_physiome_v7/selection_lock.json",
        "reports/imwut_2026/ssl_mechanism_development_v2/selection_lock.json"]
    lock = {"status": "FROZEN_PREOUTCOME_INFERENCE_CODE_LOCK", "protocol_id": "neurostress_ssl_external_v1",
        "participant_count": len(records), "window_counts": {t: len(w) for t,w in records.items()},
        "model_count": len(specs), "score_count": 3, "threshold_source": "frozen WESAD calibration, alpha 0.05",
        "parallel_workers": PARALLEL_WORKERS, "torch_threads_per_worker": 1,
        "annotations_or_task_data_read": False, "target_retraining_normalization_calibration": False,
        "fault_repeatability_pass": True, "all_source_sha256": {p: sha(ROOT/p) for p in source_paths},
        "checkpoint_sha256": {str(s["job"].get("run", s["job"].get("fold"))): s["job"]["checkpoint_sha256"] for s in specs},
        "calibration_sha256": {str(s["job"].get("run", s["job"].get("fold"))): s["job"]["calibration_sha256"] for s in specs},
        "limitations": ["E4 CSV headers cannot detect internal dropped rows", "Synthetic structured faults only"]}
    lock_path = OUTPUT / "execution_lock.json"
    OUTPUT.mkdir(mode=0o700, parents=True, exist_ok=True)
    if "--lock-only" in sys.argv:
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    else:
        if not lock_path.is_file() or json.loads(lock_path.read_text()) != lock:
            raise ValueError("Local execution lock differs from the recomputed frozen source/model lock")
        if not DEPOSIT_RECEIPT.is_file():
            raise ValueError("Public pre-score evaluator deposit receipt is required")
        receipt = json.loads(DEPOSIT_RECEIPT.read_text())
        if receipt.get("status") != "PUBLIC_PREOUTCOME_EVALUATOR_LOCK_VERIFIED":
            raise ValueError("Public evaluator/lock deposit receipt is not verified")
        if receipt.get("evaluator_git_blob_sha") != git_blob_sha(Path(__file__)) or receipt.get("lock_git_blob_sha") != git_blob_sha(lock_path):
            raise ValueError("Local evaluator or execution lock differs from the public pre-score deposit")
        parallelism = json.loads(PARALLELISM_AMENDMENT.read_text())
        if parallelism.get("status") != "PUBLIC_PREANALYSIS_PERFORMANCE_ONLY_AMENDMENT_VERIFIED":
            raise ValueError("Public performance-only parallelism amendment is required")
    if "--lock-only" in sys.argv:
        print(json.dumps({"status":"PREOUTCOME_CODE_LOCK_READY_FOR_PUBLIC_DEPOSIT", "participants":len(records),
            "total_windows":sum(map(len,records.values())), "models":len(specs)}, indent=2), flush=True)
        return
    print(json.dumps({"status":"PUBLIC_LOCK_AND_AMENDMENT_VERIFIED_READY_FOR_INFERENCE", "participants":len(records),
        "total_windows":sum(map(len,records.values())), "models":len(specs), "parallel_workers":PARALLEL_WORKERS}, indent=2), flush=True)
    all_rows = []
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print(f"SCORING_FROZEN_MODELS workers={PARALLEL_WORKERS} cpu_threads_per_worker=1 total={len(specs)}", flush=True)
    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS, initializer=_initialize_worker,
                             initargs=(records,)) as pool:
        futures = {pool.submit(_infer_worker, spec): spec for spec in specs}
        done = 0
        for future in as_completed(futures):
            spec = futures[future]
            rows = future.result()
            all_rows.extend(rows)
            done += 1
            print(f"FROZEN_MODEL_COMPLETE {done}/{len(specs)} {spec['job'].get('fold')} seed={spec['job'].get('seed')} scores={spec['scores']}", flush=True)
    temp = BASE / "participant_model_scores.csv.partial"
    fields = sorted({k for row in all_rows for k in row})
    with temp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_rows)
    temp.replace(OUTCOMES)
    summary = analyze(all_rows)
    summary["status"] = "FROZEN_INFERENCE_ANALYSIS_COMPLETE"
    summary["outcomes_sha256"] = sha(OUTCOMES)
    (BASE / "participant_level_primary_analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status":"FROZEN_INFERENCE_COMPLETE","rows":len(all_rows),"outcomes_sha256":sha(OUTCOMES),
        "any_ssl_joint_success":summary["any_ssl_joint_success"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
