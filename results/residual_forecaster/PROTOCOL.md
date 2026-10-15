# Residual (change) forecaster: prespecified protocol

Written on 2026-10-07 22:57, before any residual model was trained or its results inspected.
Starting point: release commit 6d95e21, raw dataset SHA-256 9d5e88addb3721903c30835dfab3bea4bf41fe9c427d90795b36ad5c290298e7.
Pipeline reused: scripts/10_downstream_reference_forecasts.py, unchanged up to the
forecasting target.

## Motivation and status

The main downstream result is that no LSTM forecaster beats persistence on the full test set, largely because
models trained on the level of the slow, quantized soil-moisture index cannot extrapolate to the post-outage
level. Forecasting the change y(t+H) - y(t) and adding it to the last observed value is a standard
parameterization for slowly varying series: a model that predicts zero change reduces to persistence.
This is a secondary analysis added after the main results were known; it does not replace any reported result,
and it is reported whatever its outcome.

## Design (only the target changes)

- Target: d = y(t+H) - y(t), where y(t) is the last value of the target channel in the 12-step input window of
  the (training-variant) stream; prediction = y_obs(t) + d_hat, with y_obs(t) the last observed test input.
- Everything else identical to the level forecaster: five training variants (raw, rules->repair,
  LSTM-AE->repair, union->repair, drop-faulty), same flags, repair, chronological split, training-only
  normalization (inputs as before; d standardized by its training mean and sd), LSTM(32)->Dense(1), 12 inputs,
  30 epochs, batch 256, validation split 0.1, early stopping (patience 5), seeds 42-51, op determinism, the
  same common untouched test sets at H = 1, 12, 72 (1,162 / 1,117 / 1,040 samples).
- No tuning: no other architecture, loss, window, epoch count or seed is tried.

## Outcomes and analysis

- Primary: 5-min horizon, full screened test set, RMSE of rules->repair (residual) vs. persistence, and
  rules->repair (residual) vs. raw (residual).
- Secondary: the other variants; inside/outside the training support; H = 1 h and 6 h; MAE; comparison of
  each residual model with the level model of the same variant.
- Reported per variant: mean +/- sd over seeds; skill relative to persistence; temporal inference on pooled
  (seed-averaged) squared losses with the moving-block bootstrap (block 24) and the run-based bootstrap and
  Diebold-Mariano with Harvey correction, as in Table 12.

## Interpretation rules (fixed now)

- A variant "beats persistence" only if its pooled RMSE is lower and both bootstrap intervals exclude zero.
- If repaired variants beat persistence and raw does not, this is reported as evidence that repaired training
  data support a forecaster with skill over persistence in this deployment, for this parameterization only.
- If no variant beats persistence, the residual results are reported as a robustness check confirming that
  persistence is hard to beat for this target.
- If raw training is as good as or better than the repaired variants, that is reported.
- Results at other horizons or in subgroups are never promoted to primary after seeing them.
