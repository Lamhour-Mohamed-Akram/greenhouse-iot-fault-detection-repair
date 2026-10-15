# Level-free change models: results

Protocol: PROTOCOL.md (written before training, with Amendment 1). Script: scripts/20_change_models.py.
Candidates (persistence, ridge, level-free LSTM) are selected on a chronological validation slice at the end of
the training period. Detection masks, autoencoder, diurnal profiles, offsets, normalization and repairs used for
fitting are computed on the earlier part of the training period only; the validation slice is masked with the
same rules but never used to fit anything. Single test evaluation. All candidates are reported in
test_metrics_all_candidates.csv; selection in selection.csv; inference in temporal_inference.csv.
The first selection, whose validation slice was not isolated from the preprocessing, is kept in
initial_selection/ for traceability and is not used.

## Selected models

| Horizon | raw | rules->repair | LSTM-AE->repair | union->repair | drop-faulty |
|---|---|---|---|---|---|
| 5 min | level-free LSTM | ridge | ridge | ridge | ridge |
| 1 h | persistence | persistence | persistence | ridge | ridge |
| 6 h | persistence | ridge | ridge | ridge | ridge |

At 1 h the validation margins are very small (persistence 0.813, rules ridge 0.816, union ridge 0.813), so the
selection between persistence and ridge is close to a tie for the repaired variants.

## Test RMSE (full screened test set; persistence 1.29 / 1.53 / 1.97)

Selected model, and in brackets the ridge candidate of the same variant (same model family, no selection).

| Horizon | raw | rules->repair | LSTM-AE->repair | union->repair | drop-faulty |
|---|---|---|---|---|---|
| 5 min | 1.22 [1.28] | **1.11** [1.11] | 1.16 [1.16] | 1.16 [1.16] | 1.11 [1.11] |
| 1 h | 1.53 pers. [1.73] | 1.53 pers. [1.36] | 1.53 pers. [1.34] | 1.34 [1.34] | 1.32 [1.32] |
| 6 h | 1.97 pers. [5.02] | 2.30 [2.30] | 2.27 [2.27] | 2.25 [2.25] | 1.88 [1.88] |

## Primary outcome (5 min, full test set)

- rules->repair (ridge) beats persistence: 1.112 vs 1.290, difference -0.178, run-based interval
  [-0.217, -0.122], moving-block [-0.198, -0.155]; skill 0.14.
- The selected raw model (level-free LSTM) also beats persistence, by less (1.221, -0.069 [-0.093, -0.053]).
- rules->repair beats raw: -0.109 [-0.125, -0.069]. Same family (ridge vs ridge): -0.173 [-0.213, -0.123].

## Secondary

- 1 h: validation selects persistence for raw, rules->repair and LSTM-AE->repair; union->repair (ridge, 1.34,
  -0.20 [-0.23, -0.09]) and drop-faulty (ridge, 1.32, -0.21 [-0.25, -0.08]) beat persistence. Within the ridge
  family every repaired variant beats persistence on the test set (1.34 to 1.36) and raw ridge does not (1.73).
- 6 h: no repaired variant beats persistence; rules, autoencoder and union ridge models are worse (2.25 to 2.30,
  intervals exclude zero); drop-faulty is slightly better (1.88, run-based interval [-0.15, +0.02] includes zero).
- Drop-faulty is competitive with repair for the linear model at every horizon (it needs few windows); it is not a
  repair result and is reported separately.

## Interpretation limits

This is an exploratory analysis added after the main results were known (third parameterization after the level
LSTM and the residual LSTM). Candidates, selection rule and outcomes were fixed before training and selection uses
training data only, but the bootstrap intervals do not account for these sequential analysis choices. The gain is
attributable to the combination of training-data treatment (repair or deletion) and model representation
(level-free inputs), not to imputation alone.
