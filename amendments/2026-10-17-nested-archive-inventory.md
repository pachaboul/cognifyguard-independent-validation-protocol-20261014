# Pre-outcome amendment: nested recording archives

- Protocol: `nurses_ssl_external_v1`
- Date: 2026-10-17 UTC
- Trigger: checksum-verified top-level archive metadata inventory found 624 recording-level ZIP entries and no direct CSV members. No nested archive has been opened; no signal values or annotations have been read.
- Amendment: before signal access, permit reading each recording-level ZIP member only as a nested archive container to inspect its central-directory metadata. Do not extract or open any inner member during this inventory. Record only hashed participant/recording path tokens, member basenames, compressed/uncompressed byte counts, and allowed-channel membership. The only eligible inner members are `ACC.csv`, `BVP.csv`, `EDA.csv`, and `TEMP.csv`. All other inner members, including `tags.csv`, `HR.csv`, `IBI.csv`, and any non-signal files, remain excluded and unopened.
- Analysis, sampling, model, threshold, fault, inclusion, and statistical rules are unchanged. If nested metadata does not provide an unambiguous mapping to the four allowed signal streams, stop and report the archive format as incompatible before reading signal values.
- This amendment is timestamped before nested archive inventory and before any signal values are read.
