# Script 14 - Equal-sensor weighting of the imputation errors and block-length sensitivity
#
#   (a) normalized imputation RMSE averaged per sensor first, then with equal weight over the
#       three sensors (humidity has two held-out windows, the other sensors one); rankings,
#       paired differences with a bootstrap stratified by sensor, and a normalized version of
#       the RMSE-vs-gap figure
#   (b) moving-block bootstrap intervals of the main downstream RMSE differences for block lengths
#       of 12, 24 and 48 samples, and a segment bootstrap that resamples whole contiguous runs of
#       retained test samples (so no block joins two non-adjacent samples); the block-24 interval and the
#       difference are those of script 10 (Table 12, same bootstrap draw), so that Supplementary Table S8
#       and Table 12 agree; the separate block-24 draw of this script is kept as b24_redraw_*
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
DATA = ROOT / "data" / "data.csv.gz"
RES = ROOT / "results"; OUT_T = RES / "tables"; OUT_F = RES / "figures"
TAB, INJ, MODELS, OFIG, ORIG_T, MS_F = OUT_T, RES / "injected", RES / "models", OUT_F, OUT_T, OUT_F
OUT_T.mkdir(parents=True, exist_ok=True); OUT_F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
SENSORS = ["humidity", "temperature", "soil_moisture"]; GAPS = [5, 15, 60, 240, 720]
rng = np.random.default_rng(0)

# ============================================================ (a) equal-sensor weighting
imp = pd.read_csv(OUT_T / "imputation_benchmark_raw.csv")
sd = json.load(open(OUT_T / "imputation_extended_headline.json"))["clean_sd"]
imp["nrmse"] = imp.apply(lambda r: r.rmse / sd[r.sensor], axis=1)
per_sensor = imp.groupby(["node_outage", "method", "gap_min", "sensor"]).nrmse.mean().reset_index()
eq = per_sensor.groupby(["node_outage", "method", "gap_min"]).nrmse.mean().unstack("gap_min").round(3)
pooled = imp.groupby(["node_outage", "method", "gap_min"]).nrmse.mean().unstack("gap_min").round(3)
eq.to_csv(OUT_T / "imputation_normalized_rmse_equal_sensor.csv")
rows = []
for outage in [False, True]:
    for L in GAPS:
        e = eq.loc[outage][L]; p = pooled.loc[outage][L]
        rows.append({"node_outage": outage, "gap_min": L, "best_equal_sensor": e.idxmin(), "best_pooled": p.idxmin(),
                     "linear_equal": float(e["linear"]), "best_equal_value": float(e.min()),
                     "linear_gap_to_best_pct": round(100 * (float(e["linear"]) / float(e.min()) - 1), 1)})
rank = pd.DataFrame(rows); rank.to_csv(OUT_T / "imputation_rank_equal_vs_pooled.csv", index=False); print(rank.to_string())

def strat_boot(a, b, outage, L, n=2000):
    """Bootstrap of the equal-sensor mean difference: resample (window, repeat) units within each sensor."""
    sub = imp[(imp.node_outage == outage) & (imp.gap_min == L)]
    piv = sub.pivot_table(index=["sensor", "window_start", "repeat"], columns="method", values="nrmse")
    d = {s: (piv.loc[s][a] - piv.loc[s][b]).values for s in SENSORS}
    point = float(np.mean([v.mean() for v in d.values()]))
    boots = [np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in d.values()]) for _ in range(n)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
ci = []
for outage in [False, True]:
    for L in GAPS:
        for a, b in [("linear", "ffill"), ("linear", "knn"), ("linear", "spline3"), ("linear", "profile")]:
            m, lo, hi = strat_boot(a, b, outage, L)
            ci.append({"node_outage": outage, "gap_min": L, "a": a, "b": b, "mean_diff": round(m, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
ci = pd.DataFrame(ci); ci.to_csv(OUT_T / "imputation_paired_ci_equal_sensor.csv", index=False); print(ci.to_string())

# repair-policy selection on the windows that precede the downstream split (two humidity windows):
# numerically best method, paired interval of every method against it, and the simplest method whose
# interval spans zero (simplicity order below)
SIMPLE = ["linear", "ffill", "spline3", "profile", "hybrid", "knn", "iterative"]
split_time = pd.Timestamp("2022-05-12 13:00")
tr = imp[pd.to_datetime(imp.window_start) < split_time]
pol = []
for outage in [False, True]:
    for L in GAPS:
        sub = tr[(tr.node_outage == outage) & (tr.gap_min == L)]
        piv = sub.pivot_table(index=["window_start", "repeat"], columns="method", values="nrmse")
        means = piv.mean(); best = means.idxmin(); tied = []
        for m in SIMPLE:
            d = (piv[m] - piv[best]).values
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            if m == best or lo <= 0 <= hi: tied.append(m)
        pol.append({"node_outage": outage, "gap_min": L, "best": best, "best_nrmse": round(float(means[best]), 3),
                    "tied_with_best": ";".join(tied), "selected": tied[0], "n_units": len(piv)})
pol = pd.DataFrame(pol); pol.to_csv(OUT_T / "imputation_policy_selection_training_windows.csv", index=False); print(pol.to_string())

# normalized RMSE-vs-gap figure (equal-sensor weighting)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, outage in zip(axes, [False, True]):
    for name in ["ffill", "linear", "spline3", "knn", "iterative", "profile", "hybrid"]:
        v = eq.loc[outage].loc[name]
        ax.plot(GAPS, v.values, "o-", ms=3, lw=1, label=name)
    ax.set_xscale("log"); ax.set_xticks(GAPS, ["5m", "15m", "1h", "4h", "12h"]); ax.set_xlabel("Gap length")
    ax.set_ylim(0, 2.6)
    ax.set_title("Single-sensor gap" if not outage else "Node outage (all sensors)", fontsize=9)
axes[0].set_ylabel("nRMSE (RMSE / sd of clean signal)\nequal weight per sensor")
axes[1].legend(fontsize=7, ncol=2)
fig.suptitle("Normalized imputation error vs. gap length (spline values above 2.6 clipped)", y=1.03)
fig.savefig(OUT_F / "fig8_imputation_rmse_vs_gap.png"); fig.savefig(OUT_F / "fig8_imputation_rmse_vs_gap.pdf"); plt.close(fig)

# ============================================================ (b) block-length sensitivity
def mbb(la, lb, block, n=2000):
    N = len(la); nb = int(np.ceil(N / block)); out = np.empty(n)
    for k in range(n):
        st = rng.integers(0, N - block + 1, nb); sel = np.concatenate([np.arange(s, s + block) for s in st])[:N]
        out[k] = np.sqrt(la[sel].mean()) - np.sqrt(lb[sel].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
def segment_boot(la, lb, ts, n=2000):
    t = pd.to_datetime(ts); breaks = np.r_[0, np.where(np.diff(t.values).astype("timedelta64[m]").astype(int) > 5)[0] + 1, len(t)]
    segs = [np.arange(breaks[i], breaks[i + 1]) for i in range(len(breaks) - 1)]
    out = np.empty(n)
    for k in range(n):
        sel = np.concatenate([segs[j] for j in rng.integers(0, len(segs), len(segs))])
        out[k] = np.sqrt(la[sel].mean()) - np.sqrt(lb[sel].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(segs)
PAIRS = [("raw", "persistence"), ("rules_repair", "persistence"), ("ae_repair", "persistence"), ("union_repair", "persistence"), ("drop_faulty", "persistence"),
         ("rules_repair", "raw"), ("ae_repair", "raw"), ("union_repair", "raw"), ("drop_faulty", "raw"), ("ae_repair", "rules_repair")]
t12 = pd.read_csv(OUT_T / "downstream_temporal_inference.csv").set_index(["H", "scope", "a", "b"])   # script 10, Table 12
brows = []
for H in [1, 12, 72]:
    z = np.load(OUT_T / f"downstream_losses_H{H}.npz"); inside = z["inside"].astype(bool); ts = z["ts"]
    for scope, sel in [("all", np.ones(len(inside), bool)), ("inside_support", inside)]:
        for a, b in PAIRS:
            la, lb = z[a][sel], z[b][sel]
            r12 = t12.loc[(H, scope, a, b)]
            assert abs(float(np.sqrt(la.mean()) - np.sqrt(lb.mean())) - r12["diff"]) < 5e-4
            row = {"H": H, "scope": scope, "a": a, "b": b, "diff": float(r12["diff"])}
            for blk in [12, 24, 48]:
                lo, hi = mbb(la, lb, blk)
                if blk == 24:   # Table 12 interval; the draw is still made so that the random stream of the other columns is unchanged
                    row["b24_lo"], row["b24_hi"] = float(r12.ci_lo), float(r12.ci_hi); row["b24_redraw_lo"] = round(lo, 3); row["b24_redraw_hi"] = round(hi, 3)
                else:
                    row[f"b{blk}_lo"] = round(lo, 3); row[f"b{blk}_hi"] = round(hi, 3)
            lo, hi, ns = segment_boot(la, lb, ts[sel]); row["seg_lo"] = round(lo, 3); row["seg_hi"] = round(hi, 3); row["n_segments"] = ns
            brows.append(row)
bl = pd.DataFrame(brows); bl.to_csv(OUT_T / "downstream_block_sensitivity.csv", index=False); print(bl.to_string())
print("done")
