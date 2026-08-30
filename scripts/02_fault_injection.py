# %% [markdown]
# # Script 02 — Controlled Fault Injection for Labeled Ground Truth
#
# Script 01 characterized the *real* faults in the deployment. Real faults have no
# ground-truth labels at the sample level, so detector evaluation (Script 03) also
# needs **controlled injection** into screened-clean segments.
#
# Protocol:
# 1. Work on the 1-min resampled grid (same basis as Notebook 01).
# 2. Select **clean windows**: contiguous stretches with no dropout, no out-of-range,
#    no stuck-at for the target sensor.
# 3. Inject six parameterized fault types at three severities, seeded for
#    reproducibility: spike, stuck-at, drift, bias, noise, dropout.
# 4. Save a long-format labeled dataset + injection manifest for Notebooks 03–04.
#
# Target sensors: `humidity`, `temperature`, `soil_moisture` — the three dynamic
# sensors that are healthy over usable stretches (CO₂ is calibration-biased,
# soil temperature is dead, light is binary).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
RESULTS = BASE / "results"
FIG, TAB = RESULTS / "figures", RESULTS / "tables"
INJ = RESULTS / "injected"
INJ.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

SEED = 42
TARGETS = ["humidity", "temperature", "soil_moisture"]
VALID_RANGE = {"humidity": (0, 100), "temperature": (-5, 55), "soil_moisture": (0, 100)}
STUCK_THRESH_MIN = 60      # same 1-hour stuck plausibility threshold as Notebook 01
WINDOW_LEN = 24 * 60       # clean-window length: 24 h of 1-min samples
MAX_WINDOWS_PER_SENSOR = 4

# %%
df = pd.read_csv(BASE / "data" / "data.csv.gz", encoding="latin-1")
df.columns = ["Id", "humidity", "temperature", "soil_moisture", "soil_temp", "co2", "light", "date"]
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df1 = df.set_index("date")[TARGETS].resample("1min").median()
print(f"1-min grid: {len(df1):,} rows, {df1.index.min()} -> {df1.index.max()}")

# %% [markdown]
# ## 1. Clean-window selection

# %%
def clean_mask(x, lo, hi):
    ok = x.notna() & (x >= lo) & (x <= hi)
    # exclude stuck stretches: value exactly constant for > STUCK_THRESH_MIN minutes
    grp = (x.diff() != 0).cumsum()
    runlen = x.groupby(grp).transform("size")
    ok &= ~((runlen > STUCK_THRESH_MIN) & x.notna())
    # exclude real spikes (Hampel, same parameters as Notebook 01) so injected
    # scenarios carry no unlabeled faults; ±30 min guard around each spike
    med = x.rolling(61, center=True, min_periods=10).median()
    mad = (x - med).abs().rolling(61, center=True, min_periods=10).median()
    spike = ((x - med).abs() > 6 * 1.4826 * mad) & mad.gt(0)
    ok &= ~spike.rolling(61, center=True, min_periods=1).max().astype(bool)
    return ok

def find_windows(ok, min_len):
    grp = (ok != ok.shift()).cumsum()
    out = []
    for _, seg in ok[ok].groupby(grp[ok]):
        if len(seg) >= min_len:
            out.append((seg.index[0], seg.index[-1], len(seg)))
    return out

windows = []
for s in TARGETS:
    ok = clean_mask(df1[s], *VALID_RANGE[s])
    segs = find_windows(ok, WINDOW_LEN)
    # split long clean segments into non-overlapping 24 h windows, keep the first few
    for start, end, n in segs:
        t = start
        while t + pd.Timedelta(minutes=WINDOW_LEN - 1) <= end:
            windows.append({"sensor": s, "start": t,
                            "end": t + pd.Timedelta(minutes=WINDOW_LEN - 1)})
            t += pd.Timedelta(minutes=WINDOW_LEN)
    kept = [w for w in windows if w["sensor"] == s][:MAX_WINDOWS_PER_SENSOR]
    windows = [w for w in windows if w["sensor"] != s] + kept
    print(f"{s}: {ok.mean() * 100:.1f}% clean minutes -> {len(kept)} windows of 24 h kept")

win_df = pd.DataFrame(windows).sort_values(["sensor", "start"]).reset_index(drop=True)
win_df.to_csv(TAB / "clean_windows.csv", index=False)
win_df

# %% [markdown]
# ## 2. Fault injectors
#
# Each injector takes a clean series and returns (faulty series, boolean label mask).
# Severities are scaled by the window's own standard deviation σ so they are
# comparable across sensors.

# %%
SEVERITIES = ["low", "medium", "high"]

def inject_spike(x, rng, sev):
    k = {"low": 3, "medium": 5, "high": 8}[sev]
    frac = {"low": 0.002, "medium": 0.005, "high": 0.01}[sev]
    y, m = x.copy(), np.zeros(len(x), bool)
    idx = rng.choice(len(x), max(1, int(frac * len(x))), replace=False)
    y.iloc[idx] += rng.choice([-1, 1], len(idx)) * k * x.std()
    m[idx] = True
    return y, m

def inject_stuck(x, rng, sev):
    dur = {"low": 90, "medium": 240, "high": 480}[sev]  # minutes (all > 1 h threshold)
    y, m = x.copy(), np.zeros(len(x), bool)
    start = rng.integers(0, len(x) - dur)
    y.iloc[start:start + dur] = y.iloc[start]
    m[start:start + dur] = True
    return y, m

def inject_drift(x, rng, sev):
    k = {"low": 1, "medium": 3, "high": 5}[sev]
    y, m = x.copy(), np.zeros(len(x), bool)
    start = rng.integers(len(x) // 4, len(x) // 2)
    ramp = np.linspace(0, k * x.std(), len(x) - start)
    y.iloc[start:] += ramp
    m[start:] = True
    return y, m

def inject_bias(x, rng, sev):
    k = {"low": 1, "medium": 2, "high": 3}[sev]
    y, m = x.copy(), np.zeros(len(x), bool)
    start = rng.integers(len(x) // 4, len(x) // 2)
    y.iloc[start:] += k * x.std() * rng.choice([-1, 1])
    m[start:] = True
    return y, m

def inject_noise(x, rng, sev):
    k = {"low": 0.5, "medium": 1.0, "high": 2.0}[sev]
    y, m = x.copy(), np.zeros(len(x), bool)
    start = rng.integers(0, len(x) // 2)
    dur = len(x) // 4
    y.iloc[start:start + dur] += rng.normal(0, k * x.std(), dur)
    m[start:start + dur] = True
    return y, m

def inject_dropout(x, rng, sev):
    dur = {"low": 15, "medium": 60, "high": 240}[sev]  # minutes
    y, m = x.copy(), np.zeros(len(x), bool)
    start = rng.integers(0, len(x) - dur)
    y.iloc[start:start + dur] = np.nan
    m[start:start + dur] = True
    return y, m

INJECTORS = {"spike": inject_spike, "stuck": inject_stuck, "drift": inject_drift,
             "bias": inject_bias, "noise": inject_noise, "dropout": inject_dropout}

# %% [markdown]
# ## 3. Build the labeled dataset

# %%
rng = np.random.default_rng(SEED)
records, manifest = [], []
scenario_id = 0
for _, w in win_df.iterrows():
    x = df1.loc[w.start:w.end, w.sensor]
    for ftype, fn in INJECTORS.items():
        for sev in SEVERITIES:
            y, m = fn(x, rng, sev)
            records.append(pd.DataFrame({
                "scenario_id": scenario_id, "sensor": w.sensor, "fault_type": ftype,
                "severity": sev, "timestamp": x.index,
                "value_clean": x.values, "value_faulty": y.values, "is_fault": m}))
            manifest.append({"scenario_id": scenario_id, "sensor": w.sensor,
                             "window_start": w.start, "fault_type": ftype,
                             "severity": sev, "n_fault_samples": int(m.sum()),
                             "fault_pct": round(100 * m.mean(), 2)})
            scenario_id += 1

data = pd.concat(records, ignore_index=True)
# Reproducibility invariants of the final injection design (9 windows x 6 fault
# types x 3 severities): fail loudly if the corpus ever drifts.
assert data["scenario_id"].nunique() == 162, data["scenario_id"].nunique()
assert len(data) == 233_280, len(data)
assert data.groupby("scenario_id").size().eq(1440).all()
data.to_parquet(INJ / "injected_scenarios.parquet", index=False)
mf = pd.DataFrame(manifest)
mf.to_csv(INJ / "injection_manifest.csv", index=False)
print(f"{scenario_id} scenarios, {len(data):,} labeled samples "
      f"({100 * data.is_fault.mean():.1f}% faulty overall)")
mf.groupby(["fault_type", "severity"]).scenario_id.count().unstack()

# %% [markdown]
# ## 4. Figure — one example per fault type

# %%
example_sensor = "temperature" if (win_df.sensor == "temperature").any() else win_df.sensor.iloc[0]
fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=False)
for ax, ftype in zip(axes.ravel(), INJECTORS):
    sc = mf[(mf.sensor == example_sensor) & (mf.fault_type == ftype)
            & (mf.severity == "medium")].scenario_id.iloc[0]
    d = data[data.scenario_id == sc]
    ax.plot(d.timestamp, d.value_clean, lw=0.7, color="tab:blue", label="clean")
    ax.plot(d.timestamp, d.value_faulty, lw=0.7, color="tab:red", alpha=0.8, label="faulty")
    yl = ax.get_ylim()
    ax.fill_between(d.timestamp, *yl, where=d.is_fault, color="tab:red", alpha=0.08)
    ax.set_ylim(yl)
    ax.set_title(f"{ftype} (medium)", fontsize=9)
    ax.tick_params(axis="x", labelsize=6)
axes[0, 0].legend(fontsize=7)
fig.suptitle(f"Injected fault types — {example_sensor}, 24 h clean window", y=1.0)
fig.tight_layout()
fig.savefig(FIG / "fig4_injected_fault_types.png")
fig.savefig(FIG / "fig4_injected_fault_types.pdf")
plt.close(fig)
print("Saved fig4_injected_fault_types")
