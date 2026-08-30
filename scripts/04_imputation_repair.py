# %% [markdown]
# # Script 04 — Repair: Gap Imputation Benchmark
#
# Once faults are detected (Script 03), the flagged samples must be repaired
# before downstream use. Repair = mask flagged samples → impute. This notebook
# benchmarks imputation methods on **controlled gaps** carved into screened-
# clean test windows, as a function of gap length.
#
# Methods:
# - **ffill** — last observation carried forward (what naive pipelines do)
# - **linear** — linear interpolation
# - **spline3** — cubic spline interpolation
# - **knn** — multivariate KNN using the co-recorded healthy sensors
# - **iterative** — multivariate iterative imputation (MICE-style, BayesianRidge)
# - **profile** — diurnal profile from clean training windows
# - **hybrid** — linear for gaps ≤ 1 h, diurnal profile for longer gaps
#   (a candidate imputer only: the final operational repair policy is selected
#   empirically from the benchmark and is not this fixed hybrid rule)
#
# Gap lengths mirror the real dropout distribution from Script 01
# (5 min … 12 h; the real deployment reached 130.8 h).

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

BASE = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
RESULTS = BASE / "results"
FIG, TAB, INJ = RESULTS / "figures", RESULTS / "tables", RESULTS / "injected"

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

SEED = 42
rng = np.random.default_rng(SEED)
TARGETS = ["humidity", "temperature", "soil_moisture"]
COVARIATES = ["humidity", "temperature", "soil_moisture", "light"]
GAP_LENGTHS_MIN = [5, 15, 60, 240, 720]
REPEATS = 5
N_TRAIN_WINDOWS = 2

# %%
raw = pd.read_csv(BASE / "data" / "data.csv.gz", encoding="latin-1")
raw.columns = ["Id", "humidity", "temperature", "soil_moisture", "soil_temp",
               "co2", "light", "date"]
raw["date"] = pd.to_datetime(raw["date"])
grid = raw.sort_values("date").set_index("date")[COVARIATES].resample("1min").median()

win_df = pd.read_csv(TAB / "clean_windows.csv", parse_dates=["start", "end"])
train_windows, test_windows = {}, {}
for s, g in win_df.groupby("sensor"):
    starts = sorted(g.start)
    n_train = max(1, min(N_TRAIN_WINDOWS, len(starts) - 1))  # keep >=1 test window
    train_windows[s] = starts[:n_train]
    test_windows[s] = starts[n_train:]
print({s: f"{len(train_windows[s])} train / {len(test_windows[s])} test"
       for s in TARGETS})

# diurnal profiles from training windows only
profiles = {}
for s in TARGETS:
    parts = []
    for t0 in train_windows[s]:
        w = grid.loc[t0:t0 + pd.Timedelta(minutes=1439), s]
        parts.append(w)
    x = pd.concat(parts)
    prof = x.groupby(x.index.hour * 60 + x.index.minute).mean()
    prof = prof.reindex(range(1440)).interpolate(limit_direction="both")
    profiles[s] = prof.rolling(31, center=True, min_periods=1).mean()

# %% [markdown]
# ## 1. Imputation methods

# %%
def impute_ffill(w, s):
    return w[s].ffill().bfill()

def impute_linear(w, s):
    return w[s].interpolate("linear", limit_direction="both")

def impute_spline3(w, s):
    try:
        return w[s].interpolate("spline", order=3, limit_direction="both")
    except Exception:
        return impute_linear(w, s)

def impute_knn(w, s):
    out = pd.DataFrame(KNNImputer(n_neighbors=10).fit_transform(w),
                       index=w.index, columns=w.columns)
    return out[s]

def impute_iterative(w, s):
    imp = IterativeImputer(max_iter=10, random_state=SEED)
    out = pd.DataFrame(imp.fit_transform(w), index=w.index, columns=w.columns)
    return out[s]

def impute_profile(w, s):
    minute = w.index.hour * 60 + w.index.minute
    prof = pd.Series(profiles[s].reindex(minute).values, index=w.index)
    # anchor the profile to the window level using the observed samples
    offset = (w[s] - prof).mean()
    return w[s].fillna(prof + offset)

def impute_hybrid(w, s):
    x = w[s]
    gap_id = x.isna().ne(x.isna().shift()).cumsum()
    gap_len = x.isna().groupby(gap_id).transform("sum").where(x.isna(), 0)
    short = impute_linear(w, s)
    long_ = impute_profile(w, s)
    return short.where(gap_len <= 60, long_)

METHODS = {"ffill": impute_ffill, "linear": impute_linear, "spline3": impute_spline3,
           "knn": impute_knn, "iterative": impute_iterative,
           "profile": impute_profile, "hybrid": impute_hybrid}

# %% [markdown]
# ## 2. Benchmark protocol
#
# For each (sensor, test window, gap length): mask `REPEATS` random contiguous
# gaps (one at a time), impute with every method, score RMSE/MAE on the masked
# samples only. Covariate columns stay intact — the multivariate methods may use
# them; in a real dropout the whole node often goes silent, so we also run a
# **node-outage** variant where all sensors are masked simultaneously.

# %%
def run_benchmark(mask_all_covariates):
    rows = []
    for s in TARGETS:
        for t0 in test_windows[s]:
            w0 = grid.loc[t0:t0 + pd.Timedelta(minutes=1439), COVARIATES].copy()
            if w0[s].isna().any():
                w0[s] = w0[s].interpolate(limit_direction="both")
            for L in GAP_LENGTHS_MIN:
                for r in range(REPEATS):
                    start = int(rng.integers(0, len(w0) - L))
                    w = w0.copy()
                    if mask_all_covariates:
                        w.iloc[start:start + L, :] = np.nan
                    else:
                        w.iloc[start:start + L, w.columns.get_loc(s)] = np.nan
                    truth = w0[s].iloc[start:start + L]
                    for name, fn in METHODS.items():
                        est = fn(w, s).iloc[start:start + L]
                        rows.append({
                            "sensor": s, "window_start": t0, "gap_min": L,
                            "repeat": r, "method": name, "node_outage": mask_all_covariates,
                            "rmse": float(np.sqrt(np.mean((est - truth) ** 2))),
                            "mae": float(np.mean(np.abs(est - truth)))})
    return pd.DataFrame(rows)

res_single = run_benchmark(mask_all_covariates=False)
res_outage = run_benchmark(mask_all_covariates=True)
res = pd.concat([res_single, res_outage], ignore_index=True)
res.to_csv(TAB / "imputation_benchmark_raw.csv", index=False)
print(f"{len(res):,} benchmark rows")

# %%
agg = (res.groupby(["node_outage", "method", "gap_min"]).rmse.mean().round(4)
       .unstack())
agg.to_csv(TAB / "imputation_rmse_by_gap.csv")
print(agg)

# %% [markdown]
# ## 3. Figures

# %%
# Fig 8 — RMSE vs gap length, both regimes
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, outage in zip(axes, [False, True]):
    for name in METHODS:
        m = (res[(res.node_outage == outage) & (res.method == name)]
             .groupby("gap_min").rmse.mean())
        ax.plot(m.index, m.values, "o-", ms=3, lw=1, label=name)
    ax.set_xscale("log")
    ax.set_xticks(GAP_LENGTHS_MIN, [f"{g}m" if g < 60 else f"{g // 60}h"
                                    for g in GAP_LENGTHS_MIN])
    ax.set_xlabel("Gap length")
    ax.set_title("Single-sensor gap" if not outage else "Node outage (all sensors)",
                 fontsize=9)
axes[0].set_ylabel("RMSE (masked samples, mean)")
axes[1].legend(fontsize=7, ncol=2)
fig.suptitle("Imputation error vs. gap length", y=1.03)
fig.savefig(FIG / "fig8_imputation_rmse_vs_gap.png")
fig.savefig(FIG / "fig8_imputation_rmse_vs_gap.pdf")
plt.close(fig)

# %%
# Fig 9 — qualitative example: a 4 h gap repaired by each method
s = next(x for x in ["temperature", "humidity", "soil_moisture"] if test_windows[x])
t0, L = test_windows[s][0], 240
w0 = grid.loc[t0:t0 + pd.Timedelta(minutes=1439), COVARIATES].copy()
w0[s] = w0[s].interpolate(limit_direction="both")
start = 600
w = w0.copy()
w.iloc[start:start + L, w.columns.get_loc(s)] = np.nan
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(w0.index, w0[s], lw=1, color="black", label="truth")
for name in ["ffill", "linear", "profile", "hybrid", "iterative"]:
    est = METHODS[name](w, s)
    seg = est.iloc[start:start + L]
    ax.plot(seg.index, seg.values, lw=1, alpha=0.9, label=name)
ax.axvspan(w.index[start], w.index[start + L - 1], color="gray", alpha=0.15)
ax.legend(fontsize=7, ncol=3)
ax.set_ylabel("Air temperature (°C)")
ax.set_title(f"Repairing a {L // 60} h gap — method comparison")
fig.savefig(FIG / "fig9_imputation_example.png")
fig.savefig(FIG / "fig9_imputation_example.pdf")
plt.close(fig)
print("Saved fig8, fig9")

# %% [markdown]
# ## 4. Headline numbers

# %%
# Tie handling: when two methods have identical RMSE within numerical tolerance
# and one is the simpler primitive embedded inside the other (linear inside the
# hybrid candidate for gaps <= 1 h), report the simpler method.
SIMPLICITY = ["linear", "ffill", "spline3", "profile", "knn", "iterative", "hybrid"]
best = {}
for outage in [False, True]:
    for L in GAP_LENGTHS_MIN:
        m = (res[(res.node_outage == outage) & (res.gap_min == L)]
             .groupby("method").rmse.mean())
        tied = m[m <= m.min() + 1e-9].index.tolist()
        winner = sorted(tied, key=SIMPLICITY.index)[0]
        key = f"{'outage' if outage else 'single'}_{L}min"
        best[key] = {"best_method": winner, "rmse": round(float(m[winner]), 4),
                     "ffill_rmse": round(float(m["ffill"]), 4)}
with open(TAB / "imputation_headline.json", "w") as f:
    json.dump(best, f, indent=2)
print(json.dumps(best, indent=2))
