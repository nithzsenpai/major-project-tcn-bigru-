# Changelog

## 1.1.0

- Added an automated, validation-only hyperparameter optimization workflow.
- Added sequence-length trials for 16, 24, and 32 messages.
- Added learning-rate trials for 0.001, 0.0005, and 0.0003.
- Added dropout trials for 0.20, 0.25, and 0.30.
- Added a safe class-weighting versus balanced-sampling comparison.
- Added an optional hierarchical Normal/Attack then Sybil/Illusion classifier.
- Added automatic multi-seed training and mean/standard-deviation reporting.
- Added profile validation for all required scenarios and attack archives.
- Added scenario-filtered preprocessing and richer dataset manifests.
- Added a `--skip-test` training mode so hyperparameter selection never reads Test.
- Added final full-profile execution after recommended-profile tuning.
- Added automated tests for optimization coverage and the end-to-end selection flow.

## 1.0.2

- Added retries for temporary Zenodo 429 and 5xx failures.
- Fixed writable NumPy handling for RSSI features on Windows/Pandas combinations.
