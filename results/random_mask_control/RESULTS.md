# Random-mask control: results

Protocol: `PROTOCOL.md` (including Amendment 1, written before any model was trained).
Script: `scripts/17_random_mask_control.py`; table and diagnostics: `scripts/18_random_control_table.py`.

## Execution and validity checks

- 120 fits completed (raw 10, rules->repair 10, primary random control 50, split control 50); 0 failed.
- raw and rules->repair reproduce the per-seed RMSE of the downstream script to 1.8e-15.
- All 120 prediction files share the same 1,162 test timestamps and targets; test inputs and targets are the
  untouched original observations; masks use training data only.
- Mask matching: requested and achieved additional counts and sorted run lengths are identical for every
  channel and seed (both controls); no placement failed and none needed the adjacency fallback.
  Final union with genuine missingness: identical counts by construction; union run counts differ by at
  most 5 runs; linear/fallback repair shares differ by at most 0.51 percentage points (`mask_diagnostics.csv`).
- Overlap of the random masks with the rules mask: air humidity 1 to 13 %, air temperature 25 to 32 %,
  soil-moisture index 83 to 85 % (primary) and 49 to 55 % (split). The soil-moisture overlap is structural:
  the 2,026-slot flagged run (the out-of-range episode) fits only in the observed stretch that contains it.

## Results (5-min horizon)

| Training data | RMSE full | RMSE inside | RMSE outside |
|---|---|---|---|
| Persistence | 1.29 | 1.34 | 1.25 |
| Raw | 10.94 +/- 3.45 | 4.91 +/- 1.79 | 13.93 +/- 4.36 |
| Rules->repair | 9.90 +/- 0.88 | 1.05 +/- 0.12 | 13.19 +/- 1.17 |
| Random-mask->repair (primary) | 7.36 +/- 1.94 | 1.29 +/- 0.73 | 9.74 +/- 2.57 |
| Random-mask->repair (split) | 6.27 +/- 2.30 | 2.05 +/- 0.96 | 8.16 +/- 2.99 |

Mean +/- sd over fits. Delta = control minus rules (positive favors rules).

- **Primary outcome (full test set):** the random control is better than rules-guided repair for all five
  mask seeds (mask-level Delta -1.33 to -3.48; mean over the 50 paired fits -2.55; 4 of 50 fits favor rules).
  Pooled-loss Delta -2.34, run-based bootstrap interval [-5.11, -0.62], moving-block intervals also exclude
  zero. The split control gives the same direction (-3.27, [-8.43, -0.13]).
- **Inside the training support (secondary):** primary control: mean Delta +0.24, pooled +0.42 [+0.31, +0.72],
  but mask-dependent (3 of 5 masks favor rules; masks 101 and 202 favor the control). Split control: rules
  better for every mask (5/5), every training seed (10/10) and 44 of 50 fits; pooled +1.20 [+0.96, +1.75].
- **Outside the support:** the control is better for every mask in both designs.

Three kinds of variability are reported separately: across mask seeds (5 values per control), across training
seeds (10 paired values) and temporal sampling of the shared test series (bootstrap on pooled losses, which
does not include mask or seed variability). Mean RMSE across fits and RMSE of pooled losses differ and are
labelled.

## Why the random control wins on the full test set

56 % of the test targets (68 to 85 index units) lie above every training value. Models trained on
rules-repaired data saturate near the top of the training range (slope of predictions on the last observed
value outside the support 0.16, maximum prediction 53); models trained with the random controls follow the
last value more closely (0.40 primary, 0.60 split; maxima 60 and 67). The random masks leave part of the
out-of-range episode in the training data (7 to 10 % primary, 48 to 53 % split); its large steps teach the
network to track its lagged input over a wide range, which helps extrapolation and hurts accuracy inside the
support (across the ten mask groups, the share of the episode left unmasked correlates +0.75 with inside-support
RMSE and -0.45 with full-set RMSE; ten points, descriptive only). Raw training shows the same tendency
(slope 0.51) together with the bias of the episode. The full-set advantage of the control is therefore an
extrapolation effect in the post-outage regime, not evidence of healthier training data.

## Verdict (per the prespecified rules)

- Primary outcome: **random masking performs better.** The control does not support a benefit of targeting
  on the full screened test set; implementation validity was checked and no error was found.
- Inside the training support, targeting the dominant soil-moisture fault matters: when the control cannot
  re-mask most of that fault (split control), rules-guided repair is better for every mask and seed. In the
  primary control, which unavoidably re-masks 90 to 93 % of it, the inside-support comparison is mixed.
- Much of the gain of any repaired variant over raw training inside the support (4.91 -> 1.05 to 2.05) comes
  from masking and repairing a comparable amount of data that includes most of the out-of-range episode.
- Nothing here bears on superiority over persistence: persistence remains better than every learned model on
  the full set.
