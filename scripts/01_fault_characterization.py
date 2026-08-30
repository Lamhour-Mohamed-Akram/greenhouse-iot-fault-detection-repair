# %% [markdown]
# # Script 01 — Fault Characterization of Raw Greenhouse IoT Data
#
# Fault Detection and Repair for Greenhouse IoT.
#
# This notebook characterizes the **real, uncurated faults** present in the raw
# dataset collected by the greenhouse IoT system (Article 1). No faults are
# injected here — everything reported comes from the deployment itself.
#
# Fault taxonomy used throughout the paper:
#
# | Class | Definition |
# |-------|------------|
# | **Out-of-range (OOR)** | Value outside the sensor's physically possible range |
# | **Spike** | Transient deviation inconsistent with local signal statistics (Hampel criterion) |
# | **Stuck-at** | Value exactly constant for longer than a plausibility threshold |
# | **Dead sensor** | Zero variance over the entire deployment |
# | **Calibration bias** | Long-run median outside the physically plausible range (systematic offset) |
# | **Dropout** | Transmission gap — no samples received for > 5 min |

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
DATA_CSV = BASE / "data" / "data.csv.gz"
RESULTS = BASE / "results"
FIG = RESULTS / "figures"
TAB = RESULTS / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

SENSORS = ["humidity", "temperature", "soil_moisture", "soil_temp", "co2", "light"]
SENSOR_LABELS = {"humidity": "Air humidity (%)", "temperature": "Air temperature (°C)",
                 "soil_moisture": "Soil moisture (%)", "soil_temp": "Soil temperature (°C)",
                 "co2": "CO₂ (ppm)", "light": "Light (binary)"}

# Point-level physically plausible ranges. CO2 lower bound is 0 (not ambient ~400)
# because the sensor carries a systematic negative bias — treated as a separate
# whole-sensor calibration fault, not as per-sample OOR.
VALID_RANGE = {"humidity": (0, 100), "temperature": (-5, 55), "soil_moisture": (0, 100),
               "soil_temp": (-5, 55), "co2": (0, 5000), "light": (0, 1)}

# Plausible range for the *long-run median* of each sensor (greenhouse conditions);
# a median outside this band indicates a systematic calibration bias.
PLAUSIBLE_MEDIAN = {"humidity": (20, 100), "temperature": (5, 45), "soil_moisture": (0, 100),
                    "soil_temp": (5, 45), "co2": (350, 2000), "light": (0, 1)}

# %%
df = pd.read_csv(DATA_CSV, encoding="latin-1")
df.columns = ["Id", "humidity", "temperature", "soil_moisture", "soil_temp", "co2", "light", "date"]
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"Rows: {len(df):,}")
print(f"Span: {df.date.min()} -> {df.date.max()} ({df.date.dt.date.nunique()} days with data)")
df[SENSORS].describe().T

# %% [markdown]
# ## 1. Dropout faults (transmission gaps)

# %%
dt = df["date"].diff().dt.total_seconds()
print(f"Median sampling interval: {dt.median():.0f} s")

GAP_THRESH_S = 300  # > 5 min without samples counts as a dropout
gap_idx = np.where(dt > GAP_THRESH_S)[0]
gaps = pd.DataFrame({
    "gap_start": df["date"].iloc[gap_idx - 1].values,
    "gap_end": df["date"].iloc[gap_idx].values,
    "duration_h": dt.iloc[gap_idx].values / 3600,
}).sort_values("duration_h", ascending=False).reset_index(drop=True)
gaps.to_csv(TAB / "dropout_gaps.csv", index=False)

total_span_h = (df.date.max() - df.date.min()).total_seconds() / 3600
missing_h = gaps["duration_h"].sum()
print(f"Dropouts >5min: {len(gaps)} | total missing: {missing_h:.1f} h "
      f"({100 * missing_h / total_span_h:.1f}% of the {total_span_h:.0f} h span)")
gaps

# %% [markdown]
# ## 2. Out-of-range faults

# %%
oor_rows = []
oor_mask = pd.DataFrame(False, index=df.index, columns=SENSORS)
for s in SENSORS:
    lo, hi = VALID_RANGE[s]
    m = (df[s] < lo) | (df[s] > hi)
    oor_mask[s] = m
    oor_rows.append({"sensor": s, "valid_min": lo, "valid_max": hi,
                     "observed_min": df[s].min(), "observed_max": df[s].max(),
                     "oor_samples": int(m.sum()), "oor_pct": 100 * m.mean()})
oor = pd.DataFrame(oor_rows)
oor.to_csv(TAB / "out_of_range.csv", index=False)
oor

# %% [markdown]
# ## 3. Stuck-at faults
#
# A sensor is flagged stuck when its value is *exactly* constant for longer than
# `STUCK_THRESH` (1 hour). At ~1 s sampling, short constant runs are expected from
# quantization; hour-long exact constancy in a living greenhouse is not.

# %%
STUCK_THRESH_S = 3600

def constant_runs(series, dates):
    # a run breaks when the value changes OR a dropout gap interrupts sampling,
    # so stuck durations never absorb transmission-gap time
    gap_break = dates.diff().dt.total_seconds() > GAP_THRESH_S
    grp = ((series.diff() != 0) | gap_break).cumsum()
    runs = pd.DataFrame({"value": series, "grp": grp, "date": dates})
    agg = runs.groupby("grp").agg(value=("value", "first"), start=("date", "first"),
                                  end=("date", "last"), n=("date", "size"))
    agg["duration_s"] = (agg["end"] - agg["start"]).dt.total_seconds()
    return agg

stuck_rows, stuck_mask = [], pd.DataFrame(False, index=df.index, columns=SENSORS)
stuck_episodes_all = {}
for s in SENSORS:
    if s == "light":  # binary sensor: constancy is normal
        stuck_rows.append({"sensor": s, "episodes": 0, "stuck_samples": 0,
                           "stuck_pct": 0.0, "longest_h": 0.0, "note": "binary, excluded"})
        continue
    agg = constant_runs(df[s], df["date"])
    ep = agg[agg["duration_s"] > STUCK_THRESH_S]
    stuck_episodes_all[s] = ep
    gap_break = df["date"].diff().dt.total_seconds() > GAP_THRESH_S
    grp = ((df[s].diff() != 0) | gap_break).cumsum()
    stuck_mask[s] = grp.isin(ep.index)
    stuck_rows.append({"sensor": s, "episodes": len(ep), "stuck_samples": int(ep["n"].sum()),
                       "stuck_pct": 100 * ep["n"].sum() / len(df),
                       "longest_h": ep["duration_s"].max() / 3600 if len(ep) else 0.0, "note": ""})
stuck = pd.DataFrame(stuck_rows)
stuck.to_csv(TAB / "stuck_at.csv", index=False)
stuck

# %% [markdown]
# ## 4. Dead sensor check

# %%
dead = pd.DataFrame({"sensor": SENSORS,
                     "n_unique": [df[s].nunique() for s in SENSORS],
                     "std": [df[s].std() for s in SENSORS],
                     "median": [df[s].median() for s in SENSORS]})
dead["dead"] = (dead["n_unique"] == 1)
dead["calibration_bias"] = [
    not (PLAUSIBLE_MEDIAN[s][0] <= df[s].median() <= PLAUSIBLE_MEDIAN[s][1])
    for s in SENSORS]
dead.to_csv(TAB / "dead_sensors.csv", index=False)
dead

# %% [markdown]
# ## 5. Spike faults (Hampel filter)
#
# A sample is a spike when it deviates from the rolling median by more than
# `N_SIGMA` scaled rolling MADs. Computed on the 1-min resampled signal so the
# window spans a meaningful physical duration, then excluded where the sample is
# already flagged OOR or stuck (fault classes are kept disjoint, priority:
# OOR > stuck > spike).

# %%
N_SIGMA, WINDOW = 6, 61  # 61-minute centered window on 1-min medians
df1 = df.set_index("date")[SENSORS].resample("1min").median()

spike_rows = {}
spike_mask_1m = pd.DataFrame(False, index=df1.index, columns=SENSORS)
for s in SENSORS:
    if s in ("light", "soil_temp"):  # binary / dead: spike test meaningless
        continue
    x = df1[s]
    med = x.rolling(WINDOW, center=True, min_periods=10).median()
    mad = (x - med).abs().rolling(WINDOW, center=True, min_periods=10).median()
    sigma = 1.4826 * mad
    spike_mask_1m[s] = ((x - med).abs() > N_SIGMA * sigma) & sigma.gt(0) & x.notna()
    spike_rows[s] = int(spike_mask_1m[s].sum())
print({k: v for k, v in spike_rows.items()})

# %% [markdown]
# ## 6. Consolidated fault taxonomy table (per-sensor, disjoint classes, 1-min basis)

# %%
oor_1m = pd.DataFrame({s: ((df1[s] < VALID_RANGE[s][0]) | (df1[s] > VALID_RANGE[s][1]))
                       for s in SENSORS}, index=df1.index)
stuck_1m = stuck_mask.set_axis(df["date"]).groupby(pd.Grouper(freq="1min")).any()
stuck_1m = stuck_1m.reindex(df1.index, fill_value=False)
missing_1m = df1["temperature"].isna()  # dropout: no samples arrived that minute

summary_rows = []
for s in SENSORS:
    oor_m = oor_1m[s] & ~missing_1m
    stk_m = stuck_1m[s] & ~oor_m & ~missing_1m
    spk_m = spike_mask_1m[s] & ~oor_m & ~stk_m & ~missing_1m
    n = len(df1)
    is_dead = bool(dead.loc[dead.sensor == s, "dead"].iloc[0])
    is_biased = bool(dead.loc[dead.sensor == s, "calibration_bias"].iloc[0])
    summary_rows.append({
        "sensor": s, "dead_sensor": is_dead, "calibration_bias": is_biased,
        "dropout_pct": 100 * missing_1m.mean(),
        "oor_pct": 100 * oor_m.sum() / n,
        "stuck_pct": 100 * stk_m.sum() / n,
        "spike_pct": 100 * spk_m.sum() / n,
        "total_faulty_pct": (100.0 if (is_dead or is_biased) else
                             100 * (missing_1m | oor_m | stk_m | spk_m).mean()),
    })
summary = pd.DataFrame(summary_rows).round(2)
summary.to_csv(TAB / "fault_taxonomy_summary.csv", index=False)
summary

# %% [markdown]
# ## 7. Figures

# %%
# Fig 1 — full-deployment overview with fault overlays
fig, axes = plt.subplots(len(SENSORS), 1, figsize=(10, 11), sharex=True)
for ax, s in zip(axes, SENSORS):
    ax.plot(df1.index, df1[s], lw=0.4, color="tab:blue")
    lo, hi = VALID_RANGE[s]
    bad = df1[s].where((df1[s] < lo) | (df1[s] > hi))
    ax.plot(df1.index, bad, ".", ms=2, color="tab:red", label="out-of-range")
    if s in spike_mask_1m and spike_mask_1m[s].any():
        ax.plot(df1.index, df1[s].where(spike_mask_1m[s]), ".", ms=3,
                color="tab:orange", label="spike")
    if stuck_1m[s].any() and s != "light":
        ax.plot(df1.index, df1[s].where(stuck_1m[s]), ".", ms=1,
                color="tab:purple", label="stuck-at")
    for _, g in gaps.iterrows():
        ax.axvspan(g.gap_start, g.gap_end, color="gray", alpha=0.35)
    ax.set_ylabel(SENSOR_LABELS[s], fontsize=7)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=6, markerscale=3)
axes[0].set_title("Raw greenhouse IoT deployment with real fault overlays "
                  "(gray = dropout gaps)")
fig.savefig(FIG / "fig1_deployment_overview_faults.png")
fig.savefig(FIG / "fig1_deployment_overview_faults.pdf")
plt.close(fig)

# %%
# Fig 2 — fault prevalence per sensor and class
plot_df = summary.set_index("sensor")[["dropout_pct", "oor_pct", "stuck_pct", "spike_pct"]]
ax = plot_df.plot(kind="bar", stacked=True, figsize=(7, 3.2),
                  color=["gray", "tab:red", "tab:purple", "tab:orange"])
ax.set_ylabel("% of deployment time (1-min basis)")
ax.set_xlabel("")
ax.legend(["Dropout", "Out-of-range", "Stuck-at", "Spike"], fontsize=8)
ax.set_title("Prevalence of real fault classes per sensor")
for i, s in enumerate(summary.sensor):
    row = summary.loc[summary.sensor == s].iloc[0]
    tag = "DEAD" if row.dead_sensor else ("BIASED" if row.calibration_bias else None)
    if tag:
        ax.annotate(tag, (i, plot_df.iloc[i].sum() + 2), ha="center",
                    color="tab:red", fontsize=8, weight="bold")
plt.gcf().savefig(FIG / "fig2_fault_prevalence.png")
plt.gcf().savefig(FIG / "fig2_fault_prevalence.pdf")
plt.close()

# %%
# Fig 3 — dropout gap durations
fig, ax = plt.subplots(figsize=(5.5, 3))
ax.bar(range(len(gaps)), gaps["duration_h"].sort_values(ascending=False), color="gray")
ax.set_yscale("log")
ax.set_xlabel("Dropout event (ranked)")
ax.set_ylabel("Duration (h, log scale)")
ax.axhline(1, color="tab:red", ls="--", lw=0.8, label="1 hour")
ax.axhline(24, color="tab:red", ls=":", lw=0.8, label="1 day")
ax.legend(fontsize=8)
ax.set_title("Transmission dropout durations")
fig.savefig(FIG / "fig3_dropout_durations.png")
fig.savefig(FIG / "fig3_dropout_durations.pdf")
plt.close(fig)

print("Figures written to", FIG)

# %% [markdown]
# ## 8. Headline numbers for the manuscript

# %%
headline = {
    "n_samples_raw": int(len(df)),
    "span_days": int((df.date.max() - df.date.min()).days),
    "dropout_events_gt5min": int(len(gaps)),
    "dropout_total_hours": round(float(missing_h), 1),
    "dropout_pct_of_span": round(float(100 * missing_h / total_span_h), 1),
    "longest_gap_hours": round(float(gaps.duration_h.max()), 1),
    "soil_moisture_oor_raw_sample_pct": round(float(oor.loc[oor.sensor == "soil_moisture", "oor_pct"].iloc[0]), 2),
    "soil_moisture_oor_time_basis_pct": round(float(summary.loc[summary.sensor == "soil_moisture", "oor_pct"].iloc[0]), 1),
    "co2_oor_raw_sample_pct": round(float(oor.loc[oor.sensor == "co2", "oor_pct"].iloc[0]), 2),
    "dead_sensors": dead.loc[dead.dead, "sensor"].tolist(),
    "suspected_calibration_bias_sensors": dead.loc[dead.calibration_bias & ~dead.dead, "sensor"].tolist(),
    "note": "raw_sample_pct uses the raw-record denominator; time_basis_pct uses the 1-min grid. calibration_bias denotes the operational screening classification (suspected), not an independently verified hardware diagnosis.",
    "co2_median_ppm": round(float(df.co2.median()), 1),
}
stuck_live = stuck[~stuck.sensor.isin(dead.loc[dead.dead, "sensor"])]
headline["stuck_longest_hours"] = round(float(stuck_live.longest_h.max()), 1)
headline["stuck_longest_sensor"] = stuck_live.loc[stuck_live.longest_h.idxmax(), "sensor"]
with open(TAB / "headline_numbers.json", "w") as f:
    json.dump(headline, f, indent=2)
print(json.dumps(headline, indent=2))
