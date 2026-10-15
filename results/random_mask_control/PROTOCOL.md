# Random-mask control: prespecified protocol

Written on 7 October 2026, before any random-mask model was trained or any of its results inspected.
Starting point: release repository commit `eaa7f96a2fdbecc3f009802fa045c14e470b9966` (clean working tree);
raw dataset `data.csv`, SHA-256 `9d5e88addb3721903c30835dfab3bea4bf41fe9c427d90795b36ad5c290298e7`.
Authoritative downstream implementation reused: `scripts/10_downstream_reference_forecasts.py`
.

## 1. Hypothesis and primary comparison

Rules-guided repair improves the downstream forecaster partly because it masks the observations the rule
detector flags, and not only because it masks and repairs a comparable amount of training data.

Primary comparison: **rules->repair vs. random-mask->repair**, 5-min horizon (H = 1), main chronological
split (first 70 % of the 5-min grid for training, boundary 12 May 2022 13:00).

Primary outcome: RMSE on the full common screened test set (1,162 samples at H = 1).
Secondary diagnostic outcomes: RMSE inside and outside the training support (0 to 50 index units, the same
definition and the same test samples as Table 11); MAE on all three populations.
References (not part of the primary comparison): raw training and persistence.

Difference convention: Delta RMSE = RMSE(random-mask->repair) - RMSE(rules->repair). Positive values favor
rules-guided repair.

## 2. Masks

For each input channel s (air humidity, air temperature, soil-moisture index), on the training partition only:

- Genuine missing mask M_s: 5-min grid slots with no observation. Identical in every condition.
- Rules mask F_s: the existing rule mask (out-of-range, partition-local raw-sample stuck runs, Hampel spikes,
  missing). F_s includes M_s.
- Additional rules mask A_s = F_s AND NOT M_s: observed values that the rules flag. Genuine missing values
  are therefore not counted twice.
- Random additional mask R_s (control): same number of observed values and the same multiset of contiguous
  run lengths as A_s, placed at random on observed training slots.
- Repair mask of the control: M_s OR R_s. Repair mask of rules->repair: M_s OR A_s (= F_s).

Eligible positions for R_s: every observed training slot (gtr[s] not missing), including naturally faulty
observed values. Faulty observations are deliberately not excluded: excluding them would make the control
preferentially damage healthy data. Random runs can therefore overlap true faults by chance; this overlap is
measured and reported.

## 3. Run-length matching and placement

- The runs of A_s are its maximal blocks of consecutive flagged observed slots on the 5-min grid.
- Each run of length L is placed as one block of L consecutive observed slots. Runs are placed in order of
  decreasing length; each start position is drawn uniformly from all feasible starts.
- A start is feasible if all L slots are observed, lie in the training partition, are not already used by a
  random run, and are not adjacent to an already placed random run (one-slot separation, so that two random
  runs never merge into a longer run). Random runs may touch genuine missing slots or the partition edge.
- Runs are never shortened, split or merged.
- If no feasible start exists for a run: first retry without the one-slot separation (documented per run);
  if still infeasible, the placement is recorded as failed for that mask seed and the seed is reported as an
  infeasible realization. It is not replaced by another seed.

Matching the additional masks does not match the final union with genuine missingness (a random run may
touch a dropout and form a longer repair gap) or the split between linear repair and the diurnal fallback.
Both are measured and reported.

Diagnostics exported per mask seed and channel: requested and achieved additional count; requested and
achieved sorted run lengths (exact equality check); overlap of R_s with A_s; coverage and run-length
summary of M_s OR R_s vs. M_s OR A_s; share of repaired slots filled by linear interpolation (repair gap of
at most 48 slots, 4 h) vs. the diurnal fallback.

## 4. Everything else held fixed

Same repair function (linear up to 48 slots, anchored diurnal profile beyond; the profile and anchoring
offsets are those already fitted on the rule-clean training data and shared by every repaired variant),
same per-variant training-only normalization, same LSTM(32)->Dense(1) model, 12-step inputs, 30 epochs,
batch 256, validation split 0.1, early stopping (patience 5, best weights restored), TensorFlow op
determinism, same common untouched test inputs and targets. No test value, test error or test-period
information is used to build any mask.

## 5. Seeds and budget

- Random-mask seeds, fixed now: 101, 202, 303, 404, 505 (channel-specific generators seeded with
  [mask_seed, channel_index]).
- Model-training seeds: 42 to 51 (as in Table 11).
- 5 x 10 = 50 random-mask fits. rules->repair and raw are retrained with the same 10 seeds in the same
  script, to save per-sample predictions; their RMSE must reproduce the existing per-seed values
  (`downstream_ref_raw.csv`, H = 1) to within 1e-6, otherwise the reuse is declared incompatible and
  reported.
- Every fit saves its predictions with timestamps; existing prediction files are reused on restart.
  Failed fits are logged with the cause and never silently dropped or substituted.

## 6. Analysis

- Per fit: RMSE and MAE on the full set, inside and outside the support.
- Mean across runs: random (50 fits) and rules (10 fits); rules results are not duplicated.
- Variation across random masks: for each mask seed m, Delta_m = mean over training seeds of
  RMSE(random, m, t) minus mean over training seeds of RMSE(rules, t); 5 values.
- Variation across training seeds: for each training seed t, Delta_t = mean over masks of RMSE(random, m, t)
  minus RMSE(rules, t); 10 values (paired on identical test targets).
- Temporal uncertainty on the shared test series: per-sample squared losses averaged over fits (random over
  all 50 fits; rules over 10), pooled RMSE = sqrt(mean pooled loss), difference with the moving-block
  bootstrap (blocks of 12, 24, 48 samples) and the run-based bootstrap over contiguous runs of retained
  samples, plus Diebold-Mariano with Harvey correction, exactly as in Table 12. The same is done per mask seed.
  These intervals capture temporal sampling variability of the shared test series given the trained models;
  they do not capture mask or training-seed variability, which are reported separately above.
- Mean RMSE across runs and RMSE of pooled losses are reported separately and labelled as such.
- A non-significant difference is reported as "no detectable difference", never as equivalence.

## 7. Interpretation rules (fixed in advance)

- Rules better on the primary outcome for all five mask seeds and with the pooled temporal interval
  excluding zero: the control supports a contribution of targeted masking in this deployment and design;
  no universal or causal claim.
- Mixed (sign depends on mask seed, or interval includes zero): report the dependence on the mask
  realization, support subgroup and matching limitations.
- No clear advantage: the evidence does not isolate a benefit of targeting beyond generic masking and repair.
- Random better: report it and check implementation validity without changing the design.
- None of these outcomes concerns superiority over persistence, which is reported only as a reference.

## Amendment 1 (written 2026-10-07 22:33, after the mask diagnostics and before any model was trained)

Mask diagnostics (`mask_diagnostics.csv`) show exact matching of counts and run lengths for every channel and
seed, with no failed or adjacent placements. They also show a structural limitation of the primary control:
the longest soil-moisture run of the rules mask (2,026 slots, the out-of-range episode of 19 to 26 April)
fits only inside the single observed stretch of 2,267 slots that contains that episode (242 feasible starts,
all in that stretch), so every random placement re-masks 83 to 85 % of the rules-flagged soil-moisture values.
For the target channel the primary control is therefore close to the rules mask by construction, and the
primary comparison mainly tests the targeting of the air-humidity and air-temperature flags (overlap with the
rules mask 1 to 13 % and 25 to 32 %) and of the short soil-moisture runs.

To make this visible, a secondary control is added and fixed now:

- **Control B (split long runs):** identical to the primary control, except that any rules run longer than
  the second-longest observed training stretch of its channel (1,737 slots) is divided into the smallest
  number of equal-length pieces that do not exceed that length, and the pieces are placed as separate runs
  under the same rules. This affects only the 2,026-slot soil-moisture run (two pieces of 1,013 slots);
  all other runs and both air channels are identical to the primary control for the same mask seed.
- Same five mask seeds, same ten training seeds (50 additional fits), same outcomes and analysis.
- Control B departs from the protocol rule "runs are never split"; it is reported as a secondary,
  transparently approximated control and never replaces the primary comparison.
- Interpretation: the primary comparison is interpreted as in Section 7. Control B is diagnostic: if the
  rules advantage is larger under B, this indicates that the primary control understates the effect of
  targeting because it cannot avoid the main fault; if not, it is reported as such.
