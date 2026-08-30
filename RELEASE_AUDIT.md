# Release audit

Reproducibility facts for this release. Verification command: `python verify_release.py`.

- Dataset: `data/data.csv.gz`, 947,682 raw records, 18 April – 22 May 2022,
  median raw interval ~1 s.
  SHA-256: `501ca3bab4f743f4123ea71ce546e9f794dea3711ef42923b98b5841e59da85b`
- Injection corpus: 162 scenarios (9 screened-clean windows x 6 fault types x
  3 severities), 233,280 labeled samples, 1,440 per scenario, seed 42.
- Detector evaluation: 72 held-out scenarios; macro-F1 (mean over scenarios):
  rules 0.444, Isolation Forest 0.333, LSTM autoencoder 0.356.
- Repair benchmark: 7 candidate methods x 5 gap lengths (5 min – 12 h) x
  2 outage regimes; selected policy: linear interpolation everywhere within the
  validated range except 12-h single-sensor gaps (KNN); nothing is validated
  beyond 12 h.
- Downstream experiment: chronological 70/30 split decided first; all fitted
  preprocessing (per-variant normalization, diurnal profiles, autoencoder
  weights and threshold) uses the training partition only; rule masks and the
  AE inference used for training repair are partition-bounded; 5 training-data
  variants differing only in training data; seeds 42-51; one common untouched
  test set of 1,012 screened-clean samples from the original observed stream;
  84.3% of 5-min rows carry at least one rule flag; 88 of 6,902 training
  windows survive deletion. TensorFlow op determinism enabled; two independent
  full runs are bitwise identical (Python 3.11.5, NumPy 2.3.4, TF 2.20.0, CPU).
- Authoritative downstream results (`results/tables/downstream_variants_summary.csv`,
  mean ± sd over 10 seeds):
  raw RMSE 9.82 ± 3.22 (R² −0.62 ± 0.97); rules→repair 7.11 ± 1.06 (+0.21);
  LSTM-AE→repair 6.28 ± 0.64 (+0.39 ± 0.12, −36.1% vs raw);
  union→repair 7.38 ± 1.06 (+0.15); drop-faulty 11.06 ± 2.12 (−0.93).
