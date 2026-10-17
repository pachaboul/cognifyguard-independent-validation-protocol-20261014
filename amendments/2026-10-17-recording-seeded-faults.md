# Pre-outcome clarification: recording-specific deterministic fault seeds

- Protocol: `nurses_ssl_external_v1`
- Date: 2026-10-17 UTC
- Trigger: eligibility preflight passed with 488 eligible recordings and one sampled window per 10-minute block. Block ordinals repeat across recordings, so the frozen structured-fault seed must include the recording ordinal.
- Exact rule: for a selected window, construct an internal fault-seed participant key `{participant_token}|recording_{recording_ordinal:04d}` and retain the zero-based block ordinal as `window_index`. For each family/modality, use the existing `structured_faults_v3|{participant_id}|{window_index}|{family}|{modality}` SHA-256 seed rule. Thus the full key is `structured_faults_v3|{participant_token}|recording_{recording_ordinal:04d}|{block_ordinal}|{family}|{modality}`. The canonical participant token remains the sole participant-level inference unit and output identifier.
- This makes fault realizations deterministic and distinct across recordings while preserving the existing fault implementation. No outcomes, model predictions, or stress/task labels have been viewed. All other protocol provisions are unchanged.
