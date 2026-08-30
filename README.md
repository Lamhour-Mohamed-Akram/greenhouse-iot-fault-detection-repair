# From Faulty Greenhouse Sensors to Trustworthy Forecasts

Code and data for the manuscript "From Faulty Greenhouse Sensors to Trustworthy Forecasts: An End-to-End Sensor Health Management Pipeline for Fault Detection, Data Repair, and Downstream Prediction".

**Status:** Submitted to the International Journal of Prognostics and Health Management (IJPHM) on 30 August 2026. Not yet accepted or published; no volume, issue, article number, or DOI exists.

Low-cost greenhouse IoT deployments produce sensor streams that machine learning models usually assume to be trustworthy. This repository contains the experiments behind the study of an uncurated 34-day deployment (947,682 raw records, six sensor streams, median interval ~1 s) in which every stream carries at least one naturally occurring fault signature: transmission dropouts covering 26.3% of the span (longest 130.8 h), an eight-day episode of physically impossible negative soil-moisture readings, a zero-variance soil-temperature channel, a suspected CO₂ calibration bias, stuck-at episodes of up to 153.1 h, and transient spikes. The pipeline characterizes those faults on a time basis, benchmarks three unsupervised detectors on 162 controlled fault-injection scenarios built from screened-clean segments of the same deployment, benchmarks seven gap-repair methods in single-sensor and whole-node outage regimes, and quantifies the downstream consequences on the deployment's own soil-moisture forecasting task under a leakage-controlled chronological protocol.

The central downstream result: training on raw data yields a forecaster worse on average than the test-set mean (R² = −0.62 ± 0.97); repairing the training data guided by LSTM-autoencoder flags reduces RMSE from 9.82 to 6.28 (36.1%) and raises R² to +0.39 ± 0.12, while deleting flagged samples is not viable because 84.3% of 5-min rows carry at least one flag, leaving 88 of 6,902 training windows (see `results/tables/downstream_variants_summary.csv`).

## Repository structure

```
scripts/     analysis pipeline: characterization, injection, detection, repair, downstream, figures
data/        raw deployment record (gzip-compressed) and provenance (see data/README.md)
results/     numerical tables, figures, and the labeled injection scenarios
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
python scripts/01_fault_characterization.py   # analysis + manuscript Figures 3–5
python scripts/02_fault_injection.py          # seeded injection experiment + manuscript Figure 2
python scripts/03_fault_detection.py          # detector benchmark + manuscript Figures 6–8
python scripts/04_imputation_repair.py        # imputation benchmark + manuscript Figures 9–10
python scripts/05b_downstream_variants.py     # downstream experiment + manuscript Figure 11
python scripts/05c_fig11.py                   # retrains two models + manuscript Figure 12
python scripts/06_architecture_figure.py      # renders the static Figure 1 schematic (no data involved)
```

Figures 2–12 are reproducible outputs computed from the supplied raw deployment data and seeded experimental procedures. Their plotted measurements, metrics, repair estimates, and model predictions are calculated by the corresponding scripts rather than entered as fixed result values. Figure 1 is a static conceptual schematic of the documented deployment architecture, drawn by script `06_architecture_figure.py` from hard-coded labels; it involves no data or experiment and is not an empirical result (the IJPHM manuscript draws the same schematic natively in TikZ). Scripts 01–05b compute the analyses from the raw data; script 05c retrains two forecasters from the raw data to draw Figure 12. The exported PNG and PDF files under `results/figures/` are included for convenient inspection and can be regenerated using the commands above.

Key protocol elements:

- **Screened-clean definition:** a 1-min slot passes the screen if it is present, in physical range, not part of a >1 h constant run, and at least 30 min from any Hampel-flagged spike (k = 6, 61-min window). Cleanliness is established by this algorithmic screen, not by external verification; injected labels, by contrast, are exact ground truth.
- **Injection (seed 42):** 6 fault types (spike, stuck, drift, bias, noise, dropout) × 3 severities × 9 windows = 162 scenarios, 233,280 labeled samples.
- **Detector split (RQ2):** the earliest clean windows per sensor train the learned detectors; evaluation uses only the 72 scenarios from held-out windows.
- **Downstream split (RQ4):** strictly chronological 70/30 on the 5-min grid; every fitted component (normalization, diurnal profiles, autoencoder weights and threshold) uses the training period only. Five training-data variants (raw, rules→repair, LSTM-AE→repair, union→repair, drop-faulty), ten seeds (42–51). The five models differ only in their training data: all are evaluated on one common untouched test set of 1,012 screened-clean samples taken from the original observed 5-min stream, never repaired or filled by any variant.

## Runtime

Scripts 01–04 take seconds to a few minutes each on a laptop CPU (script 03 trains three small LSTM autoencoders, a few minutes). Script 05b trains 50 small LSTM forecasters plus three autoencoders, roughly 30–60 minutes of CPU. All derived outputs are shipped under `results/`, so every manuscript table and figure can be inspected without rerunning anything.

## Reproducibility notes

This repository contains the authoritative leakage-controlled 10-seed downstream analysis. The ten seeds are fixed (42–51); every run is reported and no seed is excluded from any aggregate. Rule masks and the autoencoder inference used for training repair are computed per chronological partition so no rolling or reconstruction window crosses the train/test boundary, and the autoencoder threshold is estimated from fully screened-clean training sequences only. TensorFlow op determinism is enabled and all RNGs are seeded: two independent full runs of script 05b produced bitwise-identical per-seed results (Python 3.11.5, NumPy 2.3.4, TensorFlow 2.20.0, CPU). `python verify_release.py` checks the release invariants (raw row count, scenario counts, seeds, and headline consistency).

## Utilities

`scripts/make_notebook.py` converts any `# %%`-annotated script into a Jupyter notebook (e.g. `python scripts/make_notebook.py scripts/02_fault_injection.py notebooks/02_fault_injection.ipynb`); it is a convenience tool and is not part of the analysis pipeline.

## Manuscript figure mapping

Repository output filenames are historical; the IJPHM manuscript numbers figures by order of appearance. Figure 1 is the programmatically rendered architecture schematic; Figures 2–12 are computed from the data and experiments by the listed generator scripts:

| Manuscript figure | Repository output | Generator |
|---|---|---|
| Fig. 1 | `fig12_architecture.*` | `06_architecture_figure.py` |
| Fig. 2 | `fig4_injected_fault_types.*` | `02_fault_injection.py` |
| Fig. 3 | `fig1_deployment_overview_faults.*` | `01_fault_characterization.py` |
| Fig. 4 | `fig2_fault_prevalence.*` | `01_fault_characterization.py` |
| Fig. 5 | `fig3_dropout_durations.*` | `01_fault_characterization.py` |
| Fig. 6 | `fig5_detection_f1_by_fault.*` | `03_fault_detection.py` |
| Fig. 7 | `fig6_detection_severity_heatmap.*` | `03_fault_detection.py` |
| Fig. 8 | `fig7_real_deployment_detection.*` | `03_fault_detection.py` |
| Fig. 9 | `fig8_imputation_rmse_vs_gap.*` | `04_imputation_repair.py` |
| Fig. 10 | `fig9_imputation_example.*` | `04_imputation_repair.py` |
| Fig. 11 | `fig10_downstream_variants.*` | `05b_downstream_variants.py` |
| Fig. 12 | `fig11_downstream_predictions.*` | `05c_fig11.py` |

## Citation

Until the manuscript is published, please cite this repository and the manuscript as:

M. A. Lamhour, M. Msalek, M. Kasbouya, S. Ardchir, and M. Azzouazi, "From Faulty Greenhouse Sensors to Trustworthy Forecasts: An End-to-End Sensor Health Management Pipeline for Fault Detection, Data Repair, and Downstream Prediction," manuscript submitted to the International Journal of Prognostics and Health Management (IJPHM), 2026. Code and data: https://github.com/Lamhour-Mohamed-Akram/greenhouse-iot-fault-detection-repair

```
@misc{lamhour2026sensorhealth,
  title={From Faulty Greenhouse Sensors to Trustworthy Forecasts: An End-to-End Sensor
         Health Management Pipeline for Fault Detection, Data Repair, and Downstream Prediction},
  author={Lamhour, Mohamed Akram and
          Msalek, Mohamed and
          Kasbouya, Mohamed and
          Ardchir, Soufiane and
          Azzouazi, Mohamed},
  year={2026},
  howpublished={\url{https://github.com/Lamhour-Mohamed-Akram/greenhouse-iot-fault-detection-repair}},
  note={Manuscript submitted to the International Journal of Prognostics and Health Management (IJPHM), 30 August 2026}
}
```

Once the paper is formally published, this entry will be replaced with the final citation and DOI.

The deployment whose raw data this study analyzes was introduced in M. Ghazouani, M. Azzouazi, and M. A. Lamhour, "A drip irrigation prediction system in a greenhouse based on long short-term memory and connected objects," Mathematical Modeling and Computing, vol. 10, no. 2, pp. 524–533, 2023, doi:10.23939/mmc2023.02.524. The same system's forecasting pipeline was audited from an interpretability (XAI) perspective in M. A. Lamhour et al., "Multi-Seed XAI Validation for Reliable LSTM Irrigation Control," IEEE Access, vol. 14, pp. 120366–120387, 2026, doi:10.1109/ACCESS.2026.3720462. Those papers assume clean inputs; the present study is the upstream sensor-health layer (fault detection → characterization → data repair → downstream forecasting reliability) and shares no experiments, figures, or results with them.

## License

Code and data: MIT (`LICENSE`).
