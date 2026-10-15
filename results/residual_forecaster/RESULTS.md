# Residual (change) forecaster: results

Protocol: PROTOCOL.md (written before training). Script: scripts/19_residual_forecaster.py. 150 fits, none failed;
test sets identical to Table 11 (1,162 / 1,117 / 1,040 samples). Pooled RMSE = RMSE of seed-averaged squared losses.

| Horizon | Variant | Level model RMSE (Table 11) | Residual model RMSE | Persistence |
|---|---|---|---|---|
| 5 min | raw | 10.94 | 1.30 | 1.29 |
| 5 min | rules->repair | 9.90 | 1.81 | 1.29 |
| 5 min | drop-faulty | 13.48 | 1.30 | 1.29 |
| 1 h | raw | 10.71 | 1.70 | 1.53 |
| 1 h | rules->repair | 10.03 | 1.66 | 1.53 |
| 6 h | raw | 12.83 | 2.55 | 1.97 |
| 6 h | rules->repair | 11.39 | 2.08 | 1.97 |

(mean over seeds)

## Primary outcome (5 min, full test set), per the prespecified rules

- No variant beats persistence. rules->repair (residual): pooled 1.86 vs 1.29, both intervals exclude zero.
- Raw training is better than rules->repair on the full set (1.295 vs 1.86): the raw-trained residual model learns
  almost no change (skill -0.004) and reduces to persistence; repaired-data residual models predict a pull toward the
  training level, which hurts after the post-outage level shift (outside support 2.25 vs 1.28).
- Rule outcome: "no variant beats persistence" and "raw is as good as or better than the repaired variants".

## Secondary outcomes (not promoted)

- The change parameterization removes most of the extrapolation failure of the level models (5-min RMSE 9.9 -> 1.8
  for rules->repair, 10.9 -> 1.3 for raw).
- Inside the training support at 5 min, rules->repair (residual) beats persistence (1.18 vs 1.34) and raw (1.31);
  intervals exclude zero. It is not better than the rules->repair level model inside the support (1.18 vs 1.06).
- At 6 h, rules->repair (residual) shows no detectable difference from persistence on the full set (2.09 vs 1.97,
  intervals include zero), beats raw (2.60, intervals exclude zero) and beats persistence inside the support
  (1.34 vs 1.58, both intervals exclude zero).
