# From Faulty Greenhouse Sensors to Usable Training Data

Code and data for the manuscript "From Faulty Greenhouse Sensors to Usable Training Data: An End-to-End Sensor Health Management Pipeline for Fault Detection, Data Repair, and Downstream Validation".

Low-cost greenhouse IoT deployments produce sensor streams that machine learning models usually assume to be trustworthy. This repository contains the experiments behind the study of an uncurated 34-day deployment (947,682 raw records, six sensor streams, median interval ~1 s) in which every stream carries at least one naturally occurring fault signature: transmission dropouts covering 26.4% of the span (longest 130.8 h), an out-of-range soil-moisture episode at the start of the deployment whose core is a 153.1-h run at the constant value −85 (20 to 26 April), a zero-variance soil-temperature channel, a suspected CO₂ calibration bias, stuck-at episodes, transient spikes, and, after the longest outage, a variance collapse of the air-temperature channel that marks a change of operating regime. The pipeline characterizes those faults on a time basis, benchmarks three unsupervised detectors on 162 controlled fault-injection scenarios built from screened-clean segments of the same deployment (with window-clustered uncertainty, threshold sensitivity and near-threshold scenarios), benchmarks seven gap-repair methods per sensor and in normalized units in single-sensor and whole-node outage regimes, and measures the downstream consequences on the deployment's own soil-moisture forecasting task against persistence and autoregressive references under a leakage-controlled chronological protocol.

The central downstream result: LSTM forecasters of the soil-moisture level do not surpass the persistence forecast for this slow, integer-quantized target at horizons of 5 min to 6 h (persistence RMSE 1.29 at 5 min vs 9.9 to 13.5 for the learned models on the full test set). A linear AR(1) reference is slightly better than persistence at 6 h only. In an exploratory analysis, level-free change models selected on a validation slice of the training period beat persistence at 5 min when trained on repaired data (RMSE 1.11 vs 1.29, skill 0.14), against 1.22 for raw training; at 1 h validation selects persistence for raw data and for rules- and autoencoder-guided repair, union-guided repair (1.34) and drop-faulty (1.32) beat persistence (1.53), and within the ridge family every repaired variant beats persistence (1.34 to 1.36) while raw training does not (1.73); at 6 h the selected repaired-data models are worse than persistence (2.25 to 2.30 vs 1.97, bootstrap intervals excluding zero) and only drop-faulty is below it (1.88; its run-based interval includes zero, while the moving-block interval of the level models, −0.177 to −0.035, would exclude it) (scripts 19 to 21). For the level models, training-data repair still matters: on test targets inside the range covered by training (a post hoc split on the true target), repaired-data models reach the persistence level at 5 min (RMSE 1.05 to 1.30 for rules-, autoencoder- and union-guided repair vs 1.34) and at 6 h their error is at most 2.02 times that of persistence (2.88 to 3.19 vs 1.58), while raw-data models are 3.7 times worse and biased. Grouped instead by the last observed value, which is known at prediction time (script 24, section j, `results/tables/downstream_support_operational.csv`), repair still reduces the inside-support error by about 4 units relative to raw training at every horizon (−4.16 and −4.01 at 5 min for rules- and autoencoder-guided repair), but at 5 min only rules-guided repair stays at the persistence level (+0.21, interval −0.02 to 0.43) and at 1 h and 6 h every repaired variant is worse than persistence; the two groupings agree on 82.1%, 77.8% and 72.7% of the test samples. Reaching persistence also depends on the diurnal fallback used for long outages (script 13), and no training-data treatment compensates for the post-outage regime change (see `results/tables/downstream_ref_summary.csv`, `downstream_support_decomposition.csv`, `downstream_temporal_inference.csv`).

Two stuck-at conventions are kept side by side. Script `05b` (outputs `downstream_variants_*`) evaluates constancy on runs of 5-min medians, which flags a healthy slow channel as stuck: 89.6% of the training-period soil-moisture rows then carry a rule flag (out-of-range, missing, stuck or spike), and 40.6% of all training-period rows, not counting the out-of-range episode, are flagged stuck; script `10` (outputs `downstream_ref_*`) evaluates constancy on the raw samples (52.4% and 3.3%) and is the authoritative downstream result. The comparison between the two is itself one of the findings.

## Repository structure

```
scripts/     analysis pipeline: characterization, injection, detection, repair, downstream, figures (01-06 core pipeline, 07-24 extended analyses and checks)
data/        raw deployment record (gzip-compressed) and provenance (see data/README.md)
results/     numerical tables, figures, the labeled injection scenarios, and the three saved LSTM-autoencoder detectors
requirements.txt
```

## Installation

```
git clone https://github.com/Lamhour-Mohamed-Akram/greenhouse-iot-fault-detection-repair
cd greenhouse-iot-fault-detection-repair
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`tensorflow` is needed for the LSTM-autoencoder detector and the downstream experiments (scripts 03 and 05b/05c); the characterization, injection, and imputation experiments run without it.

## Data setup

The raw record ships with the repository, distributed gzip-compressed as `data/data.csv.gz` (5.6 MB rather than 62 MB); pandas reads it directly, so nothing needs unpacking and no Git LFS is used. The file is uncurated by design: it contains the naturally occurring faults that are the object of study. Do not clean it before running the pipeline. Provenance and column descriptions are in `data/README.md`.

## Reproducing the experiments

Run from the repository root, in order:

```bash
python scripts/01_fault_characterization.py   # RQ1 characterization (manuscript Figure 4, Table 7)
python scripts/02_fault_injection.py          # seeded injection corpus (Supplementary Figure S1)
python scripts/03_fault_detection.py          # detector benchmark, saves the three LSTM-AE models (Table 8, Supplementary Figure S2)
python scripts/04_imputation_repair.py        # imputation benchmark (Supplementary Figure S3; native-unit version of Figure 7)
python scripts/05b_downstream_variants.py     # downstream experiment with the grid-based stuck rule (kept for comparison)
python scripts/05c_fig11.py                   # prediction figure for the grid-based variant
python scripts/06_architecture_figure.py      # static schematic (the manuscript draws Figure 1 in TikZ)
python scripts/07_regimes_and_definitions.py  # operating regimes, variance collapse, bounds table, dropout bases, CO2 day/night on the raw-sample and 1-min time bases, per regime (Figure 2, Figure 5, Tables 4-5)
python scripts/08_detection_sensitivity.py    # window-clustered F1, paired differences, false-positive rates, threshold sensitivity, near-threshold and quasi-stuck scenarios, native amplitudes (Table 8, Supplementary Tables S1 and S3)
python scripts/09_imputation_per_sensor.py    # per-sensor and normalized RMSE, bootstrap CIs, standardized KNN (Supplementary Table S4); its first policy selection, on window-pooled nRMSE, is superseded by script 14 and kept only as imputation_policy_pooled_superseded.csv
python scripts/10_downstream_reference_forecasts.py       # downstream experiment with the raw-sample stuck rule, 3 horizons, persistence/AR(1)/seasonal references, inside/outside-support decomposition, block bootstrap + Diebold-Mariano, long-gap ablation (Figure 8, Supplementary Figure S4, Tables 11-13)
python scripts/11_figures_overview_and_real_stream.py          # Figures 3 and 6 (deployment overview, detectors on the real stream with the raw-sample stuck rule)
python scripts/12_rolling_origin_and_screens.py  # clean-screen margin sensitivity, autoencoder-based independent screen, rolling-origin blocks and operational test set (Supplementary Table S5)
python scripts/13_longgap_ablation_inside.py  # long-gap ablation with errors on the full test set and inside the training support (Table 13)
python scripts/14_weighting_and_blocks.py     # equal-sensor normalized imputation errors and paired intervals, traceable repair-policy selection (Tables 9-10, Figure 7); block-length and run-based bootstrap sensitivity of the downstream intervals
python scripts/15_window_length_sensitivity.py  # window-length sensitivity of the three detectors (autoencoder sequence, Hampel window, Isolation Forest rolling window)
python scripts/16_knn_tie_sensitivity.py      # sensitivity of the KNN imputer to distance ties (quantized channels)
python scripts/17_random_mask_control.py      # random-mask control for rules-guided repair (protocol and outputs in results/random_mask_control/; MASKS_ONLY=1 builds the masks only)
python scripts/18_random_control_table.py     # summary table and extrapolation diagnostics of the random-mask control (Supplementary Table S6)
python scripts/19_residual_forecaster.py      # residual LSTM: change target, absolute level still among the inputs (Supplementary Table S7)
python scripts/20_change_models.py            # level-free change models (persistence, ridge, level-free LSTM) selected on a validation slice of the training period (Table 14)
python scripts/21_change_model_tables.py      # tables of scripts 19 and 20
python scripts/22_supplementary_tables.py      # Supplementary Tables S2, S4, S8 and S9 (detector window lengths, per-sensor imputation errors of all seven methods, block-length sensitivity of the temporal intervals, inside-support errors grouped by the target and by the last observed input)
python scripts/23_severity_heatmap.py          # Supplementary Figure S2 redrawn from the saved per-severity F1 table
python scripts/25_predictions_figure.py        # Figure 8 (raw and rules-repaired, seed 42) redrawn from the saved predictions, no training
python scripts/24_revision_checks.py           # checks of the second revision round, no training: persistence on the submitted (1,012) and revised (1,162) test sets, stuck-rule and rule-flag shares, flagged-run lengths behind the RQ4 repair, CO2 on the time basis per regime, per-window detector differences, inside-support decomposition grouped by the last observed input with the Table 12 bootstrap (revision_checks.json, detection_paired_differences_by_window.csv, downstream_support_operational.csv)
```

Scripts 08, 11 and 12 load the LSTM-autoencoder detectors saved by script 03 under `results/models/` (shipped), so that the detection numbers of scripts 03 and 08 agree exactly (macro-F1 0.444 / 0.333 / 0.356). Scripts 10, 12 and 13 take about 25, 20 and 10 min on a laptop CPU; both are deterministic (TensorFlow op determinism, seeds 42-51).

Key protocol elements:

- **Stuck-at rule:** exact constancy of consecutive raw samples for more than 60 min, runs split at dropout boundaries, labelled retrospectively over the whole run and mapped to the 1-min and 5-min grids. Evaluating constancy on resampled medians instead over-flags slow, integer-quantized channels; script 05b keeps that convention for comparison, scripts 10-12 use the raw-sample rule.
- **Screened-clean definition:** a 1-min slot passes the screen if it is present, in physical range, not part of a >1 h constant run of 1-min medians, and at least 30 min from any Hampel-flagged spike (k = 6, 61-min window). Cleanliness is established by this algorithmic screen, not by external verification; injected labels, by contrast, are exact ground truth. Sensitivity to k (4/6/8) and to the margin (15/30/60 min), and an autoencoder-based screen independent of the rule family, are in script 08 and 12 outputs.
- **Injection (seed 42):** 6 fault types (spike, stuck, drift, bias, noise, dropout) × 3 severities × 9 windows = 162 scenarios, 233,280 labeled samples.
- **Detector split (RQ2):** the earliest clean windows per sensor train the learned detectors; evaluation uses only the 72 scenarios from held-out windows.
- **Downstream split (RQ4):** strictly chronological 70/30 on the 5-min grid (boundary 12 May 13:00); every fitted component (normalization, diurnal profiles, autoencoder weights and threshold) uses the training period only. Five training-data variants (raw, rules→repair, LSTM-AE→repair, union→repair, drop-faulty), ten seeds (42–51), three horizons (5 min, 1 h, 6 h). The five models differ only in their training data: all are evaluated on one common untouched test set (1,162 screened-clean samples at the 5-min horizon with the raw-sample stuck rule; 1,012 with the grid-based rule of script 05b) taken from the original observed 5-min stream, never repaired or filled by any variant, against persistence, seasonal-naive and AR(1) references on the same samples. Errors are decomposed by whether the target lies inside the training support (0-50 index units); inference uses a moving-block bootstrap and Diebold-Mariano tests on paired per-sample losses. Script 12 repeats the experiment on three rolling-origin blocks and on an operational test set without the rule screen.

## Runtime

Scripts 01–04, 07, 09, 11, 14 and 22 to 24 take seconds to a few minutes each on a laptop CPU. Scripts 05b, 10 and 12 each train on the order of 150 small LSTM forecasters plus autoencoders, roughly 20–60 minutes of CPU each; script 08 a few minutes. All derived outputs are shipped under `results/`, so every manuscript table and figure can be inspected without rerunning anything.

## Reproducibility notes

This repository contains the leakage-controlled 10-seed downstream analysis. The ten seeds are fixed (42–51); every run is reported and no seed is excluded from any aggregate. Rule masks and the autoencoder inference used for training repair are computed per chronological partition so no rolling or reconstruction window crosses the train/test boundary, and the autoencoder threshold is estimated from fully screened-clean training sequences only. TensorFlow op determinism is enabled and all RNGs are seeded: two independent full runs of script 05b produced bitwise-identical per-seed results (Python 3.11.5, NumPy 2.3.4, TensorFlow 2.20.0, CPU). `python verify_release.py` checks the release invariants (raw row count, scenario counts, seeds, and headline consistency).

## Utilities

`scripts/make_notebook.py` converts any `# %%`-annotated script into a Jupyter notebook (e.g. `python scripts/make_notebook.py scripts/02_fault_injection.py notebooks/02_fault_injection.ipynb`); it is a convenience tool and is not part of the analysis pipeline.

## Manuscript figure mapping

Repository output filenames are historical; the manuscript and its Supplementary Material number figures by order of appearance. Figure 1 is drawn natively in the manuscript (script 06 renders the same schematic); all other figures are computed from the data and experiments by the listed generator scripts:

| Figure | Repository output | Generator |
|---|---|---|
| Fig. 1 | `fig12_architecture.*` | `06_architecture_figure.py` |
| Fig. 2 | `fig_regimes.*` | `07_regimes_and_definitions.py` |
| Fig. 3 | `fig1_deployment_overview_faults.*` | `11_figures_overview_and_real_stream.py` |
| Fig. 4 | `fig3_dropout_durations.*` | `01_fault_characterization.py` |
| Fig. 5 | `fig2_fault_prevalence.*` | `07_regimes_and_definitions.py` |
| Fig. 6 | `fig7_real_deployment_detection.*` | `11_figures_overview_and_real_stream.py` |
| Fig. 7 | `fig8_imputation_rmse_vs_gap.*` | `14_weighting_and_blocks.py` (normalized; script 04 draws the native-unit version) |
| Fig. 8 | `fig11_downstream_predictions.*` | `25_predictions_figure.py` (from the predictions saved by `17_random_mask_control.py`; `10_downstream_reference_forecasts.py` draws the same variant) |
| Fig. S1 | `fig4_injected_fault_types.*` | `02_fault_injection.py` |
| Fig. S2 | `fig6_detection_severity_heatmap.*` | `23_severity_heatmap.py` (from the table written by `03_fault_detection.py`) |
| Fig. S3 | `fig9_imputation_example.*` | `04_imputation_repair.py` |
| Fig. S4 | `fig12_support_decomposition.*` | `10_downstream_reference_forecasts.py` |

`fig5_detection_f1_by_fault.*` and `fig10_downstream_variants.*` are produced by scripts 03 and 05b/10 but are not used in the manuscript (their content is in Tables 8 and 11).

## Citation

Please cite this repository and the manuscript as:

M. A. Lamhour, M. Msalek, M. Kasbouya, S. Ardchir, and M. Azzouazi, "From Faulty Greenhouse Sensors to Usable Training Data: An End-to-End Sensor Health Management Pipeline for Fault Detection, Data Repair, and Downstream Validation," manuscript, 2026. Code and data: https://github.com/Lamhour-Mohamed-Akram/greenhouse-iot-fault-detection-repair

```
@misc{lamhour2026sensorhealth,
  title={From Faulty Greenhouse Sensors to Usable Training Data: An End-to-End Sensor
         Health Management Pipeline for Fault Detection, Data Repair, and Downstream Validation},
  author={Lamhour, Mohamed Akram and
          Msalek, Mohamed and
          Kasbouya, Mohamed and
          Ardchir, Soufiane and
          Azzouazi, Mohamed},
  year={2026},
  howpublished={\url{https://github.com/Lamhour-Mohamed-Akram/greenhouse-iot-fault-detection-repair}},
  note={Manuscript}
}
```

Once the paper is formally published, this entry will be replaced with the final citation and DOI.

The deployment whose raw data this study analyzes was introduced in M. Ghazouani, M. Azzouazi, and M. A. Lamhour, "A drip irrigation prediction system in a greenhouse based on long short-term memory and connected objects," Mathematical Modeling and Computing, vol. 10, no. 2, pp. 524–533, 2023, doi:10.23939/mmc2023.02.524. The same system's forecasting pipeline was audited from an interpretability (XAI) perspective in M. A. Lamhour et al., "Multi-Seed XAI Validation for Reliable LSTM Irrigation Control," IEEE Access, vol. 14, pp. 120366–120387, 2026, doi:10.1109/ACCESS.2026.3720462. Those papers assume clean inputs; the present study is the upstream sensor-health layer (fault characterization → detection → data repair → downstream validation) and shares no experiments, figures, or results with them.

## License

Code and data: MIT (`LICENSE`).
