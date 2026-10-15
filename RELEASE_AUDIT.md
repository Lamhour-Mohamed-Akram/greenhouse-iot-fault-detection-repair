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
- Downstream results with the grid-based stuck rule, script 05b (`results/tables/downstream_variants_summary.csv`,
  mean ± sd over 10 seeds):
  raw RMSE 9.82 ± 3.22 (R² −0.62 ± 0.97); rules→repair 7.11 ± 1.06 (+0.21);
  LSTM-AE→repair 6.28 ± 0.64 (+0.39 ± 0.12, −36.1% vs raw);
  union→repair 7.38 ± 1.06 (+0.15); drop-faulty 11.06 ± 2.12 (−0.93).
  These are the submitted results and are superseded: the grid-based rule
  flags the healthy soil-moisture channel as stuck, and the 36.1% reduction
  is withdrawn. With the raw-sample rule (script 10, 1,162 test samples)
  LSTM-AE→repair is 11.01 vs raw 10.94 on the full test set (no gain) and
  rules→repair 9.90 (−9.5%); inside the training support rules→repair is
  1.05 vs raw 4.91 (−79%) with the samples grouped by the true target, and 1.56
  vs 5.72 (persistence 1.35) grouped by the last observed input, which is known
  at prediction time (script 24, section j). Persistence is 1.29 on both test sets (script 24).

- Extended analyses, scripts 07-12: stuck-at runs evaluated on the raw samples
  (`results/tables/downstream_ref_*`). Rules flag 51.6% of
  the real soil-moisture stream (the same 25,475 one-minute slots as the
  characterization total of Table 7; script 24);
  61.3% of 5-min rows carry at least one flag; 1,741 of 6,902 training windows
  survive deletion; common test set 1,162 samples at 5 min (1,117 at 1 h,
  1,040 at 6 h). Persistence RMSE 1.29 / 1.53 / 1.97 at 5 min / 1 h / 6 h;
  level LSTMs 9.9-14.5 on the full test set (no level LSTM beats
  persistence; see scripts 19-20 for the change models); inside the training support at 5 min: persistence 1.34,
  rules->repair 1.05, LSTM-AE->repair 1.17, union 1.30, raw 4.91, drop-faulty
  2.62. Detection macro-F1 reproduced exactly from the saved models (rules
  0.444, Isolation Forest 0.333, LSTM-AE 0.356); imputation benchmark reproduced
  to 1e-14 from a clean checkout in the pinned environment (requirements.txt;
  pandas 2.3.3, NumPy 2.3.4, scikit-learn 1.7.2, SciPy 1.16.3, macOS arm64).
  KNN results are sensitive to the software environment: the channels are
  quantized, so the 10th and 11th nearest donors are equidistant in 83% of
  single-sensor gap rows and the selected donors depend on tie-breaking order;
  breaking ties at random changed the equal-sensor KNN nRMSE by at most 0.025
  and no ranking in the environment above (script 16); this is an empirical
  result for that environment, not a bound guaranteed across environments, and
  exact reproduction is claimed only for the pinned environment. Rolling-origin blocks (5-8 May, 8-11 May, 12-22 May): persistence is best
  on the full test sets and rules->repair beats raw inside the training support in every block and
  test set; other orderings vary by block (script 12). Long-gap ablation inside
  the support: 1.05/1.17 (diurnal fallback), 1.55/1.54 (no fill beyond 4 h),
  2.97/2.39 (minus >12 h outage windows), raw 4.91 (script 13). Imputation
  errors are reported as equal-sensor normalized RMSE (script 14).

- Random-mask control (scripts 17-18, protocol written before training in
  results/random_mask_control/PROTOCOL.md): 120 fits, none failed; raw and
  rules->repair reproduce script 10 to 1.8e-15. Full screened test set: the
  random control is better than rules->repair for all five mask seeds (pooled
  difference -2.34); inside the training support rules->repair is better than
  the split control for all five mask seeds and all ten training seeds when
  averaged over the other factor (44 of 50 individual fits; +1.20 pooled) and mask-dependent
  against the primary control (+0.42 pooled, 3 of 5 masks). See RESULTS.md.

- Residual LSTM (script 19, 150 fits) and level-free change models (script 20,
  validation-based selection, 210 prediction files), both with protocols written
  before training. Selected model for rules->repair at 5 min: ridge, RMSE 1.112
  vs persistence 1.290; raw: level-free LSTM, 1.221. At 1 h validation selects
  persistence for raw, rules->repair and LSTM-AE->repair, ridge for union->repair
  (1.338) and drop-faulty (1.322). Validation candidates are fitted with rule masks,
  autoencoder, profiles and repairs learned before the validation cut only
  (Amendment 1 of the protocol); the first selection is kept in initial_selection/.
  Rerunning script 20 from the repository after deleting every cached prediction
  (all validation and test models retrained) reproduces selection.csv,
  test_metrics_all_candidates.csv, temporal_inference.csv and all 210 prediction
  files byte for byte.
  The .npz files under results/*/predictions are the per-fit test predictions;
  scripts 17, 19 and 20 reuse them instead of retraining when present.

- Second revision round (script 24, no training, `results/tables/revision_checks.json`):
  persistence RMSE 1.29 on the submitted 1,012-sample test set (grid-based stuck
  rule) and on the revised 1,162-sample set; soil-moisture stuck flags on 1-min slots
  45.3% (median rule) vs 21.5% (raw rule) of all slots counting the out-of-range run,
  26.7% vs 2.9% of all slots not counting it; total rule-flag share of 5-min training
  rows 89.6% vs 52.4%; stuck flags alone 40.6% vs 3.3% of all training rows, not
  counting the out-of-range episode. Script 14 now takes the
  block-24 column of Supplementary Table S8 from the Table 12 draw of script 10.
- Operational support split (script 24, section j, no training,
  `results/tables/downstream_support_operational.csv`): the inside/outside-support
  decomposition of Table 12 groups test samples by the true target; grouped by the
  last observed input y(t) instead (agreement 82.1%, 77.8%, 72.7% at 5 min, 1 h,
  6 h), repair still beats raw training inside the support by about 4 units at every
  horizon (5 min: rules −4.16, autoencoder −4.01, intervals excluding zero), but
  reaching persistence holds only for rules-guided repair at 5 min (+0.21, interval
  −0.02 to 0.43; autoencoder +0.35, union +0.51, intervals excluding zero), and every
  repaired variant is worse than persistence at 1 h and 6 h. The same bootstrap
  stream under the target grouping reproduces Table 12 exactly (asserted).
