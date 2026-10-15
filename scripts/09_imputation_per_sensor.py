# Script 09 - Imputation benchmark: per-sensor and normalized errors
#
# Reuses the benchmark table of script 04.
#   - per-sensor RMSE/MAE and a normalized RMSE (RMSE / sd of the clean signal)
#   - bootstrap confidence intervals on paired method differences
#   - repair-policy selection restricted to windows that precede the downstream split (superseded by script 14)
#   - KNN / iterative imputation with per-window standardization, same gap positions as script 04
#   - imputer parameters written to the headline file
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
DATA = ROOT / "data" / "data.csv.gz"
RES = ROOT / "results"; OUT_T = RES / "tables"; OUT_F = RES / "figures"
TAB, INJ, MODELS, OFIG, ORIG_T, MS_F = OUT_T, RES / "injected", RES / "models", OUT_F, OUT_T, OUT_F
OUT_T.mkdir(parents=True, exist_ok=True); OUT_F.mkdir(parents=True, exist_ok=True)
SEED = 42; TARGETS = ["humidity", "temperature", "soil_moisture"]; COVARIATES = ["humidity", "temperature", "soil_moisture", "light"]
GAPS = [5, 15, 60, 240, 720]; REPEATS = 5; N_TRAIN = 2
RQ4_SPLIT = pd.Timestamp("2022-05-12 13:00")

imp = pd.read_csv(TAB / "imputation_benchmark_raw.csv", parse_dates=["window_start"])
data = pd.read_parquet(INJ / "injected_scenarios.parquet")
win_df = pd.read_csv(TAB / "clean_windows.csv", parse_dates=["start", "end"])
train_w, test_w = {}, {}
for s, g in win_df.groupby("sensor"):
    st = sorted(g.start); n = max(1, min(N_TRAIN, len(st) - 1)); train_w[s] = st[:n]; test_w[s] = st[n:]

# sd of the clean held-out signal per sensor (over its held-out windows), for normalization
clean_sd = {}
for s in TARGETS:
    vals = []
    for t0 in test_w[s]:
        d = data[(data.sensor == s) & (data.fault_type == "spike") & (data.severity == "low")]
        d = d[pd.DatetimeIndex(d.timestamp) >= t0]; d = d[pd.DatetimeIndex(d.timestamp) < t0 + pd.Timedelta(days=1)]
        vals.append(d.value_clean.values)
    clean_sd[s] = float(np.concatenate(vals).std())
print("clean held-out sd:", clean_sd)
imp["nrmse"] = imp.apply(lambda r: r.rmse / clean_sd[r.sensor], axis=1)

# ---------------------------------------------------------------- per-sensor tables
per = imp.groupby(["node_outage", "sensor", "gap_min", "method"]).rmse.mean().unstack("method").round(3)
per["best"] = per.idxmin(axis=1); per.to_csv(OUT_T / "imputation_per_sensor_rmse.csv")
per_mae = imp.groupby(["node_outage", "sensor", "gap_min", "method"]).mae.mean().unstack("method").round(3)
per_mae.to_csv(OUT_T / "imputation_per_sensor_mae.csv")
norm = imp.groupby(["node_outage", "method", "gap_min"]).nrmse.mean().unstack("gap_min").round(3)
norm.to_csv(OUT_T / "imputation_normalized_rmse.csv")
native = imp.groupby(["node_outage", "method", "gap_min"]).rmse.mean().unstack("gap_min").round(3)
# rank consistency: best method per (regime, gap) under native mean, normalized mean, and per sensor
rank_rows = []
for outage in [False, True]:
    for L in GAPS:
        sub = imp[(imp.node_outage == outage) & (imp.gap_min == L)]
        rank_rows.append({"node_outage": outage, "gap_min": L,
                          "best_native_mean": sub.groupby("method").rmse.mean().idxmin(),
                          "best_normalized_mean": sub.groupby("method").nrmse.mean().idxmin(),
                          **{f"best_{s}": sub[sub.sensor == s].groupby("method").rmse.mean().idxmin() for s in TARGETS},
                          "linear_vs_best_native_pct": round(100 * float(sub.groupby("method").rmse.mean()["linear"] / sub.groupby("method").rmse.mean().min() - 1), 1)})
ranks = pd.DataFrame(rank_rows); ranks.to_csv(OUT_T / "imputation_rank_consistency.csv", index=False); print(ranks)

# ---------------------------------------------------------------- bootstrap CIs on paired differences (units: window x repeat)
rng = np.random.default_rng(0); ci_rows = []
for outage in [False, True]:
    for L in GAPS:
        sub = imp[(imp.node_outage == outage) & (imp.gap_min == L)]
        piv = sub.pivot_table(index=["sensor", "window_start", "repeat"], columns="method", values="nrmse")
        for a, b in [("linear", "ffill"), ("linear", "knn"), ("linear", "profile"), ("linear", "spline3"), ("knn", "ffill")]:
            d = (piv[a] - piv[b]).values; boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            ci_rows.append({"node_outage": outage, "gap_min": L, "a": a, "b": b, "n_units": len(d), "mean_diff_nrmse": round(float(d.mean()), 3),
                            "ci_lo": round(float(np.percentile(boots, 2.5)), 3), "ci_hi": round(float(np.percentile(boots, 97.5)), 3),
                            "a_better_share": round(float((d < 0).mean()), 2)})
ci = pd.DataFrame(ci_rows); ci.to_csv(OUT_T / "imputation_paired_ci.csv", index=False); print(ci)

# ---------------------------------------------------------------- policy selection on training-period windows only
# SUPERSEDED: this first selection averages nRMSE over all windows pooled (humidity, with two held-out windows, counts
# twice) and picks the numerically best method without paired intervals. Table 10 of the manuscript comes from
# script 14 (equal-sensor weighting, simplest method not significantly different from the best:
# imputation_policy_selection_training_windows.csv and imputation_rank_equal_vs_pooled.csv). The output of this
# block is kept for traceability under a name that marks it as superseded.
tr_only = imp[imp.window_start < RQ4_SPLIT]
print("windows before RQ4 split:", sorted(tr_only.window_start.unique()))
pol_rows = []
for outage in [False, True]:
    for L in GAPS:
        m_all = imp[(imp.node_outage == outage) & (imp.gap_min == L)].groupby("method").nrmse.mean()
        m_tr = tr_only[(tr_only.node_outage == outage) & (tr_only.gap_min == L)].groupby("method").nrmse.mean()
        pol_rows.append({"node_outage": outage, "gap_min": L, "best_all_windows": m_all.idxmin(), "best_training_period_windows": m_tr.idxmin(),
                         "linear_nrmse_train_only": round(float(m_tr["linear"]), 3), "ffill_nrmse_train_only": round(float(m_tr["ffill"]), 3),
                         "knn_nrmse_train_only": round(float(m_tr["knn"]), 3)})
pol = pd.DataFrame(pol_rows); pol.to_csv(OUT_T / "imputation_policy_pooled_superseded.csv", index=False); print(pol)

# ---------------------------------------------------------------- KNN / iterative with standardization , same gap positions as script 04
raw = pd.read_csv(DATA, encoding="latin-1"); raw.columns = ["Id", "humidity", "temperature", "soil_moisture", "soil_temp", "co2", "light", "date"]
raw["date"] = pd.to_datetime(raw["date"]); grid = raw.sort_values("date").set_index("date")[COVARIATES].resample("1min").median()
def knn_raw(w, s): return pd.DataFrame(KNNImputer(n_neighbors=10).fit_transform(w), index=w.index, columns=w.columns)[s]
def iter_raw(w, s): return pd.DataFrame(IterativeImputer(max_iter=10, random_state=SEED).fit_transform(w), index=w.index, columns=w.columns)[s]
def _std(w, fn, s):
    sc = StandardScaler().fit(w); z = pd.DataFrame(sc.transform(w), index=w.index, columns=w.columns)
    out = fn(z); return pd.Series(sc.inverse_transform(out)[:, w.columns.get_loc(s)], index=w.index)
def knn_std(w, s): return _std(w, lambda z: KNNImputer(n_neighbors=10).fit_transform(z), s)
def iter_std(w, s): return _std(w, lambda z: IterativeImputer(max_iter=10, random_state=SEED).fit_transform(z), s)
def linear(w, s): return w[s].interpolate("linear", limit_direction="both")
METHODS = {"linear": linear, "knn": knn_raw, "knn_std": knn_std, "iterative": iter_raw, "iterative_std": iter_std}
rng = np.random.default_rng(SEED)   # identical draw sequence to script 04 (one integer per gap, same loop order)
def run(mask_all):
    rows = []
    for s in TARGETS:
        for t0 in test_w[s]:
            w0 = grid.loc[t0:t0 + pd.Timedelta(minutes=1439), COVARIATES].copy()
            if w0[s].isna().any(): w0[s] = w0[s].interpolate(limit_direction="both")
            for L in GAPS:
                for r in range(REPEATS):
                    start = int(rng.integers(0, len(w0) - L)); w = w0.copy()
                    if mask_all: w.iloc[start:start + L, :] = np.nan
                    else: w.iloc[start:start + L, w.columns.get_loc(s)] = np.nan
                    truth = w0[s].iloc[start:start + L]
                    for name, fn in METHODS.items():
                        est = fn(w, s).iloc[start:start + L]
                        rows.append({"sensor": s, "window_start": t0, "gap_min": L, "repeat": r, "method": name, "node_outage": mask_all,
                                     "rmse": float(np.sqrt(np.mean((est - truth) ** 2))), "mae": float(np.mean(np.abs(est - truth)))})
    return pd.DataFrame(rows)
std_res = pd.concat([run(False), run(True)], ignore_index=True)
std_res["nrmse"] = std_res.apply(lambda r: r.rmse / clean_sd[r.sensor], axis=1)
std_res.to_csv(OUT_T / "imputation_standardized_raw.csv", index=False)
# consistency check: knn / iterative / linear must match script 04 exactly
chk = std_res[std_res.method.isin(["linear", "knn", "iterative"])].merge(imp, on=["sensor", "window_start", "gap_min", "repeat", "method", "node_outage"], suffixes=("_new", "_old"))
print("reproduction max |delta rmse| (should be ~0):", float((chk.rmse_new - chk.rmse_old).abs().max()))
std_summary = std_res.groupby(["node_outage", "method", "gap_min"]).nrmse.mean().unstack("gap_min").round(3)
std_summary.to_csv(OUT_T / "imputation_standardized_summary.csv"); print(std_summary)
std_per = std_res.groupby(["node_outage", "sensor", "gap_min", "method"]).rmse.mean().unstack("method").round(3)
std_per.to_csv(OUT_T / "imputation_standardized_per_sensor.csv")

params = {"ffill": "pandas ffill then bfill", "linear": "pandas interpolate(method='linear', limit_direction='both')",
          "spline3": "pandas interpolate(method='spline', order=3)", "knn": "sklearn KNNImputer(n_neighbors=10, metric='nan_euclidean', weights='uniform'), fitted on the 24-h window itself (retrospective, uses rows on both sides of the gap), covariates in native units (no standardization); a standardized variant is benchmarked alongside",
          "iterative": "sklearn IterativeImputer(estimator=BayesianRidge(), max_iter=10, tol=1e-3, initial_strategy='mean', imputation_order='ascending', random_state=42), fitted on the 24-h window itself",
          "profile": "minute-of-day mean of the training windows (31-min smoothed), anchored by the mean offset between observed samples of the window and the profile",
          "hybrid": "linear for gaps <= 60 min, anchored profile otherwise", "covariates": "humidity, temperature, soil moisture, light (co-recorded healthy channels)",
          "fitting_scope": "all imputers are fitted per 24-h held-out window; they may use observations after the gap (retrospective repair, as stated in Section 3.5)"}
with open(OUT_T / "imputation_extended_headline.json", "w") as f: json.dump({"clean_sd": clean_sd, "imputer_params": params}, f, indent=2)
print("done")
