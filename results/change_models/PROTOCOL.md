# Level-free change models with validation-based selection: prespecified protocol

Written on 2026-10-07 23:15, before any of these models was trained or evaluated on the test set.
This is the third forecasting parameterization tried in this study (after the level LSTM of Table 11 and the
residual LSTM of results/residual_forecaster/), and it is declared as the last one. All three are reported.

## Motivation (from training data only)

On rule-clean training data the 5-min change of the soil-moisture index is zero in 96 % of steps and has a
lag-1 autocorrelation of -0.22 (quantization flip-flop); a linear model on level-free inputs reduces the change
RMSE by about 10 % at 5 min, 1 % at 1 h and 9 % at 6 h relative to persistence on a held-out part of the
training period. The residual LSTM of the previous analysis failed after the level shift because its inputs
contain the absolute level. The models below receive no absolute soil-moisture level.

## Candidates (fixed)

For every training variant (raw, rules->repair, LSTM-AE->repair, union->repair, drop-faulty) and horizon
H in {5 min, 1 h, 6 h}, prediction = y(t) + predicted change, with y(t) the last observed target value:

1. Persistence (predicted change 0).
2. Ridge regression on level-free features of the 12-step input window: soil-moisture changes over the last
   1, 2, 3, 6 and 11 steps; last air humidity and air temperature; their changes over the window; sine and
   cosine of the hour of day. Features standardized on the fitting data; regularization alpha chosen from
   {0.1, 1, 10, 100, 1000} on the validation set.
3. Residual LSTM with level-free inputs: the 12-step window with the soil-moisture channel replaced by its
   difference to the last value of the window, air humidity and temperature unchanged; LSTM(32)->Dense(1),
   change target standardized, same training settings as Table 11.

## Selection (no test data)

- Validation set: the last 20 % of the training windows (chronological), restricted to windows whose 12 inputs
  and target are rule-clean in the original training stream (same definition as the test set).
- Fitting for validation: first 80 % of the training windows of the variant. LSTM validation uses seeds 42-44.
- For each variant and horizon, the candidate with the lowest validation RMSE is selected (ties to the simpler:
  persistence, then ridge, then LSTM). The selected candidate is refitted on the full training partition
  (LSTM: seeds 42-51) and evaluated once on the test set.
- All candidates are also evaluated on the test set and reported, but only the selected one enters the
  primary comparison.

## Outcomes

- Primary: 5-min horizon, full screened test set, selected model of rules->repair vs. persistence, and vs. the
  selected model of raw.
- Secondary: other variants and horizons; inside/outside the training support; MAE.
- Inference as in Table 12: pooled (seed-averaged) squared losses, moving-block bootstrap (block 24),
  run-based bootstrap, Diebold-Mariano with Harvey correction.

## Interpretation (fixed now)

- "Beats persistence" requires lower pooled RMSE and both bootstrap intervals excluding zero.
- If the selected rules->repair model beats persistence and the selected raw model does not, this is reported as
  evidence that repaired training data support a forecaster with skill over persistence, for this deployment and
  this model family.
- If persistence is selected or no model beats it on the test set, persistence is reported as the practical
  ceiling for this target; no further model family is tried.

## Amendment 1 (written 2026-10-08 14:31, before rerunning the selection)

A check of the code found that the validation step was not isolated from the learned preprocessing: rule masks,
the LSTM autoencoder, the diurnal profiles and the repaired series used to fit the validation candidates were
computed on the whole training partition, including the validation slice. This is not leakage from the test
set, but it weakens the claim that selection used an independent validation set. Correction (same design, no
other change): the validation fit partition is the training period before the validation cut; rule masks,
autoencoder (weights and threshold), profiles, offsets and repairs are computed on that fit partition only, and
the rule mask of the validation slice is computed on the validation slice only; validation windows lie entirely
inside the validation slice. After selection, the selected candidate is refitted on the full training partition
with the full-training preprocessing, as before. The first selection is kept in initial_selection/ and the
corrected selection replaces it, whatever its outcome.

Interpretation note: this analysis is exploratory. The move from the level model to the residual and then the
level-free parameterization was guided by results on the same test set, so the test set is not independent of
the research path, and the reported intervals do not account for these sequential choices.
