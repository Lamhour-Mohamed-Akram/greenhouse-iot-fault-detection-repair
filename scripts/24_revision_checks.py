# Script 24 - Checks for the second revision round (no model training, deterministic)
#
# Recomputes, from the raw data and from saved outputs only, the numbers quoted in the revised response
# letter, manuscript and supplement that no earlier script writes out:
#   (a) persistence on the submitted 1,012-sample test set (grid-based stuck rule of script 05b) and on the
#       revised 1,162-sample test set (raw-sample stuck rule of script 10)
#   (b) stuck-rule and total rule-flag shares of the soil-moisture channel on the 1-min grid and on the
#       5-min training rows, with and without the out-of-range episode; overlap of the rule coverage of the
#       real stream with the Table 7 total
#   (c) training target rows in flagged runs repaired linearly (<= 4 h), in runs of 4-12 h and in runs
#       longer than 12 h (rules mask of script 10)
#   (d) CO2 statistics on the raw-sample basis and on the 1-min time basis, per operating regime, night/day
#   (e) per-window paired F1 differences of the detectors (from detection_sensitivity_raw.csv)
#   (f) ratios and percentages derived from the saved downstream tables
#   (g) the soil-moisture out-of-range episode; (h) clean screen with the raw-sample constancy criterion;
#   (i) repair-policy selection on the pre-boundary soil-moisture windows of that screen (imputers only, no training)
#   (j) inside/outside-support decomposition of the downstream test set grouped by the last observed input instead of
#       the true target, with the moving-block bootstrap of Table 12 (saved per-sample losses only, no training)
# Functions are copied from scripts 08 and 10 (and 05b for the grid rule) without change of definition.
# Outputs: results/tables/revision_checks.json, results/tables/detection_paired_differences_by_window.csv,
# results/tables/imputation_policy_soil_raw_screen.csv, results/tables/imputation_benchmark_soil_raw_screen.csv,
# results/tables/downstream_support_operational.csv
import json
from math import comb
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
DATA = ROOT / "data" / "data.csv.gz"
RES = ROOT / "results"; OUT_T = RES / "tables"
SENSORS = ["humidity", "temperature", "soil_moisture", "soil_temp", "co2", "light"]
FEATURES = ["humidity", "temperature", "soil_moisture"]; TARGET = "soil_moisture"
VALID = {"humidity": (0, 100), "temperature": (-5, 55), "soil_moisture": (0, 100)}
FREQ = "5min"; SEQ = 12; LINEAR_MAX = 48; RUN_12H = 144     # 5-min rows: 4 h = 48, 12 h = 144

raw = pd.read_csv(DATA, encoding="latin-1"); raw.columns = ["Id"] + SENSORS + ["date"]
raw["date"] = pd.to_datetime(raw["date"]); raw = raw.sort_values("date").reset_index(drop=True)
out = {}

# ================================================================ (a) persistence on the submitted and revised test sets
grid = raw.set_index("date")[FEATURES].resample(FREQ).median()
split = int(0.7 * len(grid)); gtr, gte = grid.iloc[:split], grid.iloc[split:]

_STUCK_CACHE = {}
def raw_stuck_window(s, t0, t1, freq):          # script 10
    key = (s, t0, t1, freq)
    if key not in _STUCK_CACHE:
        r = raw[(raw["date"] >= t0) & (raw["date"] < t1)]
        x = r[s]; d = r["date"]; gap = d.diff().dt.total_seconds() > 300
        g = ((x.diff() != 0) | gap).cumsum(); st = d.groupby(g).transform("first"); en = d.groupby(g).transform("last")
        _STUCK_CACHE[key] = pd.Series(((en - st).dt.total_seconds() > 3600).values, index=d).groupby(pd.Grouper(freq=freq)).any()
    return _STUCK_CACHE[key]

def rule_parts(x, s, mode="raw"):               # script 10 rule_mask, returned by component; mode "grid" = script 05b rule
    lo, hi = VALID[s]; oor = ((x < lo) | (x > hi)) & x.notna()
    if mode == "raw":
        stuck = raw_stuck_window(s, x.index[0], x.index[-1] + pd.Timedelta(FREQ), FREQ).reindex(x.index, fill_value=False) & x.notna()
    else:
        grp = (x.diff() != 0).cumsum(); runlen = x.groupby(grp).transform("size")
        stuck = (runlen > 12) & x.notna()
    med = x.rolling(61, center=True, min_periods=10).median()
    mad = (x - med).abs().rolling(61, center=True, min_periods=10).median()
    spike = ((x - med).abs() > 6 * 1.4826 * mad) & mad.gt(0)
    return {"oor": oor, "stuck": stuck, "spike": spike, "missing": x.isna()}
def rule_mask(x, s, mode="raw"):
    p = rule_parts(x, s, mode); return p["oor"] | p["stuck"] | p["spike"] | p["missing"]

y_all = grid[TARGET].to_numpy(dtype="float32")
def persistence_on(mode):
    masks = pd.concat([pd.DataFrame({s: rule_mask(gtr[s], s, mode) for s in FEATURES}),
                       pd.DataFrame({s: rule_mask(gte[s], s, mode) for s in FEATURES})])
    ok_row = ~masks.any(axis=1).values
    ends = np.arange(SEQ, len(grid))                                     # target row i, inputs i-12..i-1 (scripts 05b and 10, H = 1)
    ok = np.array([ok_row[i - SEQ:i + 1].all() for i in ends])
    tidx = ends[(ends >= split) & ok]
    yt, pers = y_all[tidx], y_all[tidx - 1]
    e = pers - yt; ts = grid.index[tidx]
    mtr = masks.iloc[:split]
    support = gtr[TARGET][~mtr[TARGET]].dropna()
    return tidx, {"n_test": int(len(tidx)), "persistence_rmse": round(float(np.sqrt(np.mean(e ** 2))), 4),
                  "persistence_mae": round(float(np.mean(np.abs(e))), 4), "first_target": str(ts.min()), "last_target": str(ts.max()),
                  "per_day": {str(k): int(v) for k, v in pd.Series(ts.date).value_counts().sort_index().items()},
                  "train_support": [float(support.min()), float(support.max())],
                  "targets_above_50_pct": round(100 * float((yt > 50).mean()), 1),
                  "targets_outside_own_support_pct": round(100 * float(((yt < support.min()) | (yt > support.max())).mean()), 1)}
tidx_sub, sub = persistence_on("grid"); tidx_rev, rev = persistence_on("raw")
assert sub["n_test"] == 1012, sub["n_test"]
assert rev["n_test"] == 1162, rev["n_test"]
bl = pd.read_csv(OUT_T / "downstream_baselines.csv")
assert abs(rev["persistence_rmse"] - bl.loc[(bl.H == 1) & (bl.variant == "persistence"), "rmse"].iloc[0]) < 1e-3
extra = np.setdiff1d(tidx_rev, tidx_sub)
out["persistence_test_sets"] = {"submitted_grid_stuck_rule": sub, "revised_raw_stuck_rule": rev,
                                "shared_targets": int(len(np.intersect1d(tidx_sub, tidx_rev))),
                                "only_in_revised": int(len(extra)), "only_in_submitted": int(len(np.setdiff1d(tidx_sub, tidx_rev))),
                                "only_in_revised_per_day": {str(k): int(v) for k, v in pd.Series(grid.index[extra].date).value_counts().sort_index().items()},
                                "only_in_revised_targets_above_50_pct": round(100 * float((y_all[extra] > 50).mean()), 1)}

# ================================================================ (b) flag shares of the soil-moisture channel
# 1-min grid, whole deployment (script 08, real-stream coverage)
def hampel_flags(x, k=6, win=61):               # script 08
    med = x.rolling(win, center=True, min_periods=10).median(); mad = (x - med).abs().rolling(win, center=True, min_periods=10).median()
    return ((x - med).abs() > k * 1.4826 * mad) & mad.gt(0) & x.notna()
def stuck_flags(x, thresh=60):                  # script 08
    grp = (x.diff() != 0).cumsum(); runlen = x.groupby(grp).transform("size"); return (runlen > thresh) & x.notna()
g1 = raw.set_index("date")[SENSORS].resample("1min").median()
sm = g1["soil_moisture"]
x = raw["soil_moisture"]; d = raw["date"]; gap = d.diff().dt.total_seconds() > 300
g = ((x.diff() != 0) | gap).cumsum(); st = d.groupby(g).transform("first"); en = d.groupby(g).transform("last")
raw_stuck = pd.Series(((en - st).dt.total_seconds() > 3600).values, index=d).groupby(pd.Grouper(freq="1min")).any().reindex(sm.index, fill_value=False) & sm.notna()
oor = ((sm < 0) | (sm > 100)) & sm.notna(); spk = hampel_flags(sm, 6); miss = sm.isna(); grid_stuck = stuck_flags(sm)
pct = lambda m: round(100 * float(m.mean()), 1)
one_min = {"n_slots": int(len(sm)),
           "stuck_median_rule_incl_oor_pct": pct(grid_stuck), "stuck_raw_rule_incl_oor_pct": pct(raw_stuck),
           "stuck_median_rule_excl_oor_pct": pct(grid_stuck & ~oor & ~miss), "stuck_raw_rule_excl_oor_pct": pct(raw_stuck & ~oor),
           "rules_total_median_rule_pct": pct(oor | grid_stuck | spk | miss), "rules_total_raw_rule_pct": pct(oor | raw_stuck | spk | miss)}
# overlap with the Table 7 total (script 01 classes: disjoint, priority OOR > stuck > spike, missing slots of the grid)
miss_t7 = g1["temperature"].isna(); oor_t7 = oor & ~miss_t7; stk_t7 = raw_stuck & ~oor_t7 & ~miss_t7; spk_t7 = spk & ~oor_t7 & ~stk_t7 & ~miss_t7
det = oor | raw_stuck | spk | miss; t7 = miss_t7 | oor_t7 | stk_t7 | spk_t7
one_min["table7_comparison"] = {"rules_flagged_slots": int(det.sum()), "table7_total_slots": int(t7.sum()), "identical_slot_sets": bool((det == t7).all()),
                                "rules_total_pct_2dp": round(100 * float(det.mean()), 2), "table7_total_pct_2dp": round(100 * float(t7.mean()), 2),
                                "hampel_flags_pct": round(100 * float(spk.mean()), 2), "spike_disjoint_table7_pct": round(100 * float(spk_t7.mean()), 2),
                                "hampel_flags_inside_oor_slots": int((spk & oor).sum()), "hampel_flags_inside_oor_pct": round(100 * float((spk & oor).mean()), 2)}
tab7 = pd.read_csv(OUT_T / "fault_taxonomy_summary.csv").set_index("sensor")
assert abs(one_min["table7_comparison"]["table7_total_pct_2dp"] - tab7.loc["soil_moisture", "total_faulty_pct"]) < 0.006
# 5-min training rows (script 10, rule masks per partition)
pr, pg = rule_parts(gtr[TARGET], TARGET, "raw"), rule_parts(gtr[TARGET], TARGET, "grid")
tot = lambda p: p["oor"] | p["stuck"] | p["spike"] | p["missing"]
five_min = {"n_train_rows": int(len(gtr)),
            "total_rule_flag_median_rule_pct": pct(tot(pg)), "total_rule_flag_raw_rule_pct": pct(tot(pr)),
            "stuck_only_excl_oor_median_rule_pct": pct(pg["stuck"] & ~pg["oor"]), "stuck_only_excl_oor_raw_rule_pct": pct(pr["stuck"] & ~pr["oor"]),
            "stuck_incl_oor_median_rule_pct": pct(pg["stuck"]), "stuck_incl_oor_raw_rule_pct": pct(pr["stuck"]),
            "oor_pct": pct(pr["oor"]), "missing_pct": pct(pr["missing"])}
rh = json.load(open(OUT_T / "downstream_ref_headline.json"))["flags"]
assert five_min["total_rule_flag_raw_rule_pct"] == rh["rules_raw_stuck_train_pct"][TARGET]
assert five_min["total_rule_flag_median_rule_pct"] == rh["rules_grid_stuck_train_pct"][TARGET]
out["flag_shares_soil_moisture"] = {"one_min_slots_whole_deployment": one_min, "five_min_training_rows": five_min,
                                    "note": "median rule = exact constancy of consecutive grid medians (> 60 one-minute slots or > 12 five-minute rows); raw rule = exact constancy of consecutive raw samples for > 1 h, split at > 5-min gaps (scripts 01, 08, 10). 'total' = out-of-range | stuck | spike | missing."}

# ================================================================ (c) training target rows by length of the flagged run (rules mask of script 10)
mtr = pd.DataFrame({s: rule_mask(gtr[s], s) for s in FEATURES})
z = np.load(RES / "random_mask_control" / "masks.npz")                  # script 17 saves the same rules mask as missing (M) | flagged (A)
for s in FEATURES:
    assert np.array_equal(mtr[s].to_numpy(), z[f"M_{s}"] | z[f"A_{s}"]), s
runs = {}
for s in FEATURES:
    m = mtr[s]; gid = m.ne(m.shift()).cumsum(); r = m.groupby(gid).agg(["first", "size"]); r = r[r["first"]]["size"]
    obs = gtr[s].notna()
    gl = m.groupby(gid).transform("size").where(m, 0)                     # length of the flagged run each row belongs to
    runs[s] = {"rows_linear_le_4h": int(r[r <= LINEAR_MAX].sum()), "rows_runs_4_to_12h": int(r[(r > LINEAR_MAX) & (r <= RUN_12H)].sum()),
               "rows_runs_gt_12h": int(r[r > RUN_12H].sum()), "runs_4_to_12h": int(((r > LINEAR_MAX) & (r <= RUN_12H)).sum()), "runs_gt_12h": int((r > RUN_12H).sum()),
               "observed_flagged_rows_4_to_12h": int((obs & (gl > LINEAR_MAX) & (gl <= RUN_12H)).sum()), "observed_flagged_rows_gt_12h": int((obs & (gl > RUN_12H)).sum())}
    runs[s]["rows_runs_4_to_12h_pct_of_train"] = round(100 * runs[s]["rows_runs_4_to_12h"] / len(gtr), 1)
    runs[s]["rows_runs_gt_12h_pct_of_train"] = round(100 * runs[s]["rows_runs_gt_12h"] / len(gtr), 1)
lg = json.load(open(OUT_T / "downstream_ref_headline.json"))["long_gap"]
assert runs[TARGET]["rows_runs_4_to_12h"] + runs[TARGET]["rows_runs_gt_12h"] == lg["train_target_rows_profile_filled_gt4h"]["rules"]
out["rq4_repair_mapping_rules_mask"] = {"per_channel": runs, "linear_max_rows": LINEAR_MAX, "rows_12h": RUN_12H, "n_train_rows": int(len(gtr)),
                                        "note": "script 10 repairs flagged runs of <= 48 five-minute rows (4 h) linearly and longer runs with the anchored diurnal profile; Table 10 validates linear (KNN for 12-h single-sensor gaps) up to 12 h (144 rows). 'rows' include genuine missing rows inside the flagged runs; 'observed_flagged_rows' count only rows with an observed value. The autoencoder and union masks of script 10 are not saved and cannot be recounted without retraining."}

# ================================================================ (d) CO2 on the raw-sample and 1-min time bases (regime boundaries of script 07)
REGIMES = [("R1", "2022-04-18", "2022-04-27 12:33"), ("R2", "2022-04-28 12:00", "2022-05-11 01:26"), ("R3", "2022-05-16 12:12", "2022-05-22 19:59")]
co2 = raw.set_index("date")["co2"]; c1 = g1["co2"].dropna()
def night_day(s):
    h = s.index.hour; return round(float(s[(h >= 22) | (h < 5)].median()), 1), round(float(s[(h >= 10) & (h < 16)].median()), 1)
n_raw, d_raw = night_day(co2); n_1m, d_1m = night_day(c1)
co2_out = {"raw_sample_basis": {"n_samples": int(len(co2)), "median": round(float(co2.median()), 1), "pct_below_150": round(100 * float((co2 < 150).mean()), 1),
                                "median_night_22_05": n_raw, "median_day_10_16": d_raw,
                                "n_days_daily_median_below_150": int((co2.resample("1D").median() < 150).sum()), "n_days_with_data": int(co2.resample("1D").median().notna().sum())},
           "time_basis_1min": {"n_slots_with_data": int(len(c1)), "median": round(float(c1.median()), 1), "pct_below_150": round(100 * float((c1 < 150).mean()), 1),
                               "median_night_22_05": n_1m, "median_day_10_16": d_1m}, "per_regime": {}}
for name, a, b in REGIMES:
    r = co2[(co2.index >= a) & (co2.index <= b)]; m = c1.loc[a:b]; nn, dd = night_day(m)
    co2_out["per_regime"][name] = {"raw_samples": int(len(r)), "raw_sample_share_pct": round(100 * len(r) / len(co2), 1), "median_1min": round(float(m.median()), 1),
                                   "median_night_22_05_1min": nn, "median_day_10_16_1min": dd, "night_minus_day_1min": round(nn - dd, 1),
                                   "pct_below_150_1min": round(100 * float((m < 150).mean()), 1)}
ce = json.load(open(OUT_T / "characterization_extended_headline.json"))["co2"]
for k_raw, k_ce in [("median", "median_all"), ("pct_below_150", "pct_below_150"), ("median_night_22_05", "median_night_22_05"), ("median_day_10_16", "median_day_10_16")]:
    assert co2_out["raw_sample_basis"][k_raw] == ce[k_ce], k_raw
out["co2"] = co2_out

# ================================================================ (e) detection: per-window paired differences (default operating points)
res = pd.read_csv(OUT_T / "detection_sensitivity_raw.csv")
DEFAULT = {"rules": "k=6", "iforest": "c=0.01", "lstm_ae": "q=0.995"}
base = res[res.apply(lambda r: r.config == DEFAULT[r.detector], axis=1)]
assert base.groupby("detector").f1.mean().round(3).to_dict() == {"iforest": 0.333, "lstm_ae": 0.356, "rules": 0.444}
boot = pd.read_csv(OUT_T / "detection_paired_differences.csv").set_index(["fault_type", "a", "b"])
rows = []
for ft in ["bias", "drift", "dropout", "noise", "spike", "stuck"]:
    piv = base[base.fault_type == ft].pivot_table(index=["window_start", "severity"], columns="detector", values="f1")
    for a, b in [("rules", "lstm_ae"), ("rules", "iforest"), ("lstm_ae", "iforest")]:
        w = (piv[a] - piv[b]).groupby(level=0).mean()                   # per-window mean difference, as in script 08
        bb = boot.loc[(ft, a, b)]
        rows.append({"fault_type": ft, "a": a, "b": b, "n_windows": int(len(w)), "mean_diff": round(float(w.mean()), 3),
                     **{f"window_{i + 1}": round(float(v), 3) for i, v in enumerate(w.values)},
                     "windows": ";".join(str(k)[:10] for k in w.index),
                     "n_a_better": int((w > 0).sum()), "n_b_better": int((w < 0).sum()), "n_tied": int((w == 0).sum()),
                     "min_diff": round(float(w.min()), 3), "max_diff": round(float(w.max()), 3),
                     "boot_ci_lo": float(bb.ci_lo), "boot_ci_hi": float(bb.ci_hi), "distinct_bootstrap_resamples": comb(2 * len(w) - 1, len(w))})
pw = pd.DataFrame(rows); pw.to_csv(OUT_T / "detection_paired_differences_by_window.csv", index=False); print(pw.to_string())
out["detection_per_window"] = {"file": "results/tables/detection_paired_differences_by_window.csv",
                               "distinct_bootstrap_resamples_4_windows": comb(7, 4),
                               "sign_counts": {f"{r.fault_type}|{r.a}-{r.b}": f"{r.n_a_better}/{r.n_b_better}/{r.n_tied}" for r in pw.itertuples()},
                               "note": "sign_counts = windows where a is better / b is better / tied, over the four held-out windows"}

# ================================================================ (f) numbers derived from the saved downstream tables
sm_ = pd.read_csv(OUT_T / "downstream_ref_summary.csv", header=[0, 1], index_col=[0, 1])
dec = pd.read_csv(OUT_T / "downstream_support_decomposition.csv").set_index(["H", "variant"])
inf = pd.read_csv(OUT_T / "downstream_temporal_inference.csv")
vh = json.load(open(OUT_T / "downstream_variants_headline.json"))
rm = lambda H, v: float(sm_.loc[(H, v), ("rmse", "mean")])
ins = lambda H, v: float(dec.loc[(H, v), "inside_rmse"])
q = lambda H, a, b, sc: inf[(inf.H == H) & (inf.a == a) & (inf.b == b) & (inf.scope == sc)].iloc[0]
derived = {"submitted_ae_repair_improvement_vs_raw_pct": vh["improvement_vs_raw_pct"]["ae_repair"],
           "submitted_best_rmse_ae_repair": vh["rmse"]["ae_repair"],
           "full_rmse_5min": {v: round(rm(1, v), 2) for v in ["raw", "rules_repair", "ae_repair", "union_repair", "drop_faulty"]},
           "rules_vs_raw_full_5min_pct": round(100 * (rm(1, "rules_repair") / rm(1, "raw") - 1), 1),
           "ae_vs_raw_full_5min_pct": round(100 * (rm(1, "ae_repair") / rm(1, "raw") - 1), 1),
           "inside_rmse_5min": {v: round(ins(1, v), 2) for v in ["persistence", "raw", "rules_repair", "ae_repair", "union_repair", "drop_faulty"]},
           "rules_vs_raw_inside_5min_pct": round(100 * (ins(1, "rules_repair") / ins(1, "raw") - 1), 1),
           "inside_rmse_6h": {v: round(ins(72, v), 3) for v in ["persistence", "rules_repair", "ae_repair", "union_repair"]},
           "inside_ratio_to_persistence_6h": {v: round(ins(72, v) / ins(72, "persistence"), 2) for v in ["rules_repair", "ae_repair", "union_repair"]},
           "table11_ae_minus_raw_mean_of_seed_rmse_5min": round(rm(1, "ae_repair") - rm(1, "raw"), 2),
           "table12_ae_minus_raw_pooled_rmse_5min": round(float(q(1, "ae_repair", "raw", "all")["diff"]), 2),
           "table12_pooled_rmse_5min": {"ae_repair": round(float(q(1, "ae_repair", "raw", "all").rmse_a), 2), "raw": round(float(q(1, "ae_repair", "raw", "all").rmse_b), 2)}}
out["derived_from_saved_outputs"] = derived

# ================================================================ (g) the soil-moisture out-of-range episode (round 3, F1)
so = raw["soil_moisture"]; o_raw = (so < 0) | (so > 100)
run85 = (pd.Timestamp("2022-04-20 02:41:06"), pd.Timestamp("2022-04-26 11:48:54"))    # longest constant run, script 07
ep_first = raw.date[o_raw].iloc[0]
after = raw[(raw.date > run85[1]) & o_raw]; ep_last = after[after.date < after.date.min() + pd.Timedelta("1D")].date.max()
pre = raw[(raw.date >= ep_first) & (raw.date < run85[0])]; pre_o = pre[o_raw.loc[pre.index]]
out["soil_moisture_oor_episode"] = {"first_oor_sample": str(ep_first), "last_oor_sample": str(ep_last),
                                    "next_oor_sample_after_episode": str(raw.date[o_raw & (raw.date > ep_last)].min()),
                                    "constant_minus85_run": [str(run85[0]), str(run85[1])],
                                    "before_run_samples": int(len(pre)), "before_run_oor_share_pct": round(100 * len(pre_o) / len(pre), 1),
                                    "before_run_oor_min": float(pre_o.soil_moisture.min()), "before_run_oor_median": float(pre_o.soil_moisture.median()),
                                    "before_run_oor_max": float(pre_o.soil_moisture.max()),
                                    "on_18_april_oor_min_median": [float(pre_o[pre_o.date.dt.day == 18].soil_moisture.min()), float(pre_o[pre_o.date.dt.day == 18].soil_moisture.median())],
                                    "in_range_values_before_run_min_max": [float(pre[~o_raw.loc[pre.index]].soil_moisture.min()), float(pre[~o_raw.loc[pre.index]].soil_moisture.max())]}

# ================================================================ (h) clean screen with the raw-sample constancy criterion (round 3, G1)
SPLIT_TIME = pd.Timestamp("2022-05-12 13:00")                                          # RQ4 boundary, script 14
def clean_mask(x, lo, hi, k=6, raw_stk=None):                                          # script 08; raw_stk replaces the median criterion
    ok = x.notna() & (x >= lo) & (x <= hi)
    if raw_stk is None:
        grp = (x.diff() != 0).cumsum(); runlen = x.groupby(grp).transform("size"); ok &= ~((runlen > 60) & x.notna())
    else:
        ok &= ~(raw_stk & x.notna())
    med = x.rolling(61, center=True, min_periods=10).median(); mad = (x - med).abs().rolling(61, center=True, min_periods=10).median()
    spike = ((x - med).abs() > k * 1.4826 * mad) & mad.gt(0)
    ok &= ~spike.rolling(61, center=True, min_periods=1).max().astype(bool); return ok
def windows_24h(ok):                                                                    # script 02 cutting (script 08 count)
    grp = (ok != ok.shift()).cumsum(); starts = []
    for _, seg in ok[ok].groupby(grp[ok]):
        t = seg.index[0]
        while t + pd.Timedelta(minutes=1439) <= seg.index[-1]: starts.append(t); t += pd.Timedelta(minutes=1440)
    return starts
def raw_stuck_1min(s, index):
    x = raw[s]; d = raw["date"]; gap = d.diff().dt.total_seconds() > 300
    g = ((x.diff() != 0) | gap).cumsum(); st = d.groupby(g).transform("first"); en = d.groupby(g).transform("last")
    return pd.Series(((en - st).dt.total_seconds() > 3600).values, index=d).groupby(pd.Grouper(freq="1min")).any().reindex(index, fill_value=False)
cs, win_raw = {}, {}
for s in FEATURES:
    x = g1[s]; lo, hi = VALID[s]; rstk = raw_stuck_1min(s, x.index)
    ok_med, ok_raw = clean_mask(x, lo, hi), clean_mask(x, lo, hi, raw_stk=rstk)
    w_med, w_raw = windows_24h(ok_med), windows_24h(ok_raw); win_raw[s] = w_raw
    pre_b = lambda w: [str(t) for t in w if t + pd.Timedelta(minutes=1439) < SPLIT_TIME]
    w_med_in = pd.Series(False, index=x.index)
    for t in w_med: w_med_in.loc[t:t + pd.Timedelta(minutes=1439)] = True
    cs[s] = {"median_criterion": {"clean_minutes_pct": round(100 * float(ok_med.mean()), 1), "n_24h_windows": len(w_med), "windows_before_boundary": pre_b(w_med)},
             "raw_sample_criterion": {"clean_minutes_pct": round(100 * float(ok_raw.mean()), 1), "n_24h_windows": len(w_raw), "windows_before_boundary": pre_b(w_raw)},
             "raw_stuck_minutes_admitted_by_median_screen": int((ok_med & rstk).sum()),
             "raw_stuck_minutes_inside_median_screen_windows": int((w_med_in & rstk & x.notna()).sum())}
cs08 = pd.read_csv(OUT_T / "clean_screen_sensitivity.csv")
for s in FEATURES:
    ref = cs08[(cs08.hampel_k == 6) & (cs08.sensor == s)].iloc[0]
    assert (cs[s]["median_criterion"]["clean_minutes_pct"], cs[s]["median_criterion"]["n_24h_windows"]) == (ref.clean_minutes_pct, ref.n_24h_windows), s
out["clean_screen_constancy_criterion"] = cs

# ================================================================ (i) repair-policy selection on the pre-boundary soil-moisture windows of the raw-sample screen (round 3, G4)
# script 04 benchmark (methods, gaps, repeats, regimes, gap seed 42) and script 14 selection rule (bootstrap seed 0); no detector or forecaster
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
COV = ["humidity", "temperature", "soil_moisture", "light"]; GAPS = [5, 15, 60, 240, 720]; REPEATS = 5; S_ = "soil_moisture"
g1c = raw.set_index("date")[COV].resample("1min").median()
wins = [pd.Timestamp(t) for t in cs[S_]["raw_sample_criterion"]["windows_before_boundary"]]
n_tr = max(1, min(2, len(wins) - 1)); tr_w, te_w = wins[:n_tr], wins[n_tr:]                # script 04 allocation: earliest windows build the profile
xp = pd.concat([g1c.loc[t:t + pd.Timedelta(minutes=1439), S_] for t in tr_w])
prof = xp.groupby(xp.index.hour * 60 + xp.index.minute).mean().reindex(range(1440)).interpolate(limit_direction="both").rolling(31, center=True, min_periods=1).mean()
def im_ffill(w, s): return w[s].ffill().bfill()
def im_linear(w, s): return w[s].interpolate("linear", limit_direction="both")
def im_spline(w, s):
    try: return w[s].interpolate("spline", order=3, limit_direction="both")
    except Exception: return im_linear(w, s)
def im_knn(w, s): return pd.DataFrame(KNNImputer(n_neighbors=10).fit_transform(w), index=w.index, columns=w.columns)[s]
def im_iter(w, s): return pd.DataFrame(IterativeImputer(max_iter=10, random_state=42).fit_transform(w), index=w.index, columns=w.columns)[s]
def im_profile(w, s):
    p = pd.Series(prof.reindex(w.index.hour * 60 + w.index.minute).values, index=w.index); return w[s].fillna(p + (w[s] - p).mean())
def im_hybrid(w, s):
    x = w[s]; gid = x.isna().ne(x.isna().shift()).cumsum(); gl = x.isna().groupby(gid).transform("sum").where(x.isna(), 0)
    return im_linear(w, s).where(gl <= 60, im_profile(w, s))
METH = {"ffill": im_ffill, "linear": im_linear, "spline3": im_spline, "knn": im_knn, "iterative": im_iter, "profile": im_profile, "hybrid": im_hybrid}
grng = np.random.default_rng(42); brows4 = []
for outage in [False, True]:
    for t in te_w:
        w0 = g1c.loc[t:t + pd.Timedelta(minutes=1439), COV].copy()
        if w0[S_].isna().any(): w0[S_] = w0[S_].interpolate(limit_direction="both")
        for L in GAPS:
            for r in range(REPEATS):
                a = int(grng.integers(0, len(w0) - L)); w = w0.copy()
                if outage: w.iloc[a:a + L, :] = np.nan
                else: w.iloc[a:a + L, w.columns.get_loc(S_)] = np.nan
                truth = w0[S_].iloc[a:a + L]
                for name, fn in METH.items():
                    est = fn(w, S_).iloc[a:a + L]
                    brows4.append({"node_outage": outage, "window_start": str(t), "gap_min": L, "repeat": r, "method": name, "rmse": float(np.sqrt(np.mean((est - truth) ** 2)))})
b4 = pd.DataFrame(brows4); sd_soil = float(pd.concat([g1c.loc[t:t + pd.Timedelta(minutes=1439), S_] for t in te_w]).std()); b4["nrmse"] = b4.rmse / sd_soil
SIMPLE = ["linear", "ffill", "spline3", "profile", "hybrid", "knn", "iterative"]; brng = np.random.default_rng(0); pol4 = []
tab10 = {(False, 5): "linear", (False, 15): "linear", (False, 60): "linear", (False, 240): "linear", (False, 720): "knn",
         (True, 5): "linear", (True, 15): "linear", (True, 60): "linear", (True, 240): "linear", (True, 720): "linear"}   # Table 10, training-period column
for outage in [False, True]:
    for L in GAPS:
        piv = b4[(b4.node_outage == outage) & (b4.gap_min == L)].pivot_table(index=["window_start", "repeat"], columns="method", values="nrmse")
        means = piv.mean(); best = means.idxmin(); tied = []
        for mth in SIMPLE:
            d = (piv[mth] - piv[best]).values; bb = [d[brng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bb, [2.5, 97.5])
            if mth == best or lo <= 0 <= hi: tied.append(mth)
        pol4.append({"node_outage": outage, "gap_min": L, "best": best, "best_nrmse": round(float(means[best]), 3), "linear_nrmse": round(float(means["linear"]), 3),
                     "tied_with_best": ";".join(tied), "selected": tied[0], "table10_selected": tab10[(outage, L)], "differs_from_table10": tied[0] != tab10[(outage, L)]})
pol4 = pd.DataFrame(pol4); pol4.to_csv(OUT_T / "imputation_policy_soil_raw_screen.csv", index=False); b4.to_csv(OUT_T / "imputation_benchmark_soil_raw_screen.csv", index=False)
print(pol4.to_string())
out["soil_policy_raw_screen"] = {"profile_windows": [str(t) for t in tr_w], "held_out_windows": [str(t) for t in te_w], "clean_sd": round(sd_soil, 3),
                                 "rows_differing_from_table10": pol4[pol4.differs_from_table10][["node_outage", "gap_min", "selected", "table10_selected"]].to_dict("records"),
                                 "file": "results/tables/imputation_policy_soil_raw_screen.csv"}

# ================================================================ (j) operational support split (no training)
# Table 12 groups the test samples by the true target (inside the training support [0, 50]), which is unknown when the
# forecast is made. Here they are also grouped by the last observed value y(t), the 5-min median at ts - H*5 min,
# which is known at prediction time. Per-sample seed-averaged squared losses come from script 10
# (downstream_losses_H*.npz). The bootstrap is block_bootstrap_ci of script 10 (block 24, 2,000 replicates, the same
# draw for both members of a pair); its default_rng(0) stream is replayed in script 10's order of horizons, pairs and
# scopes, once per split, so the target split reproduces Table 12 (downstream_temporal_inference.csv) exactly.
SUPPORT = (0.0, 50.0); BLOCK, N_BOOT = 24, 2000
VARIANTS = ["raw", "rules_repair", "ae_repair", "union_repair", "drop_faulty"]
MODELS_J = ["persistence"] + VARIANTS

def block_bootstrap_ci(l_a, l_b, rng, block=BLOCK, n_boot=N_BOOT):     # script 10, generator passed in
    n = len(l_a); nb = int(np.ceil(n / block)); diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        sel = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        diffs[b] = np.sqrt(l_a[sel].mean()) - np.sqrt(l_b[sel].mean())
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

rmse_m = lambda l, m: float(np.sqrt(l[m].mean())) if m.any() else None
t12 = pd.read_csv(OUT_T / "downstream_temporal_inference.csv")
y5 = grid[TARGET]; jrows, jsum = [], {}
for split_name in ["target", "last_input"]:
    rng_j = np.random.default_rng(0)
    for H in [1, 12, 72]:
        z = np.load(OUT_T / f"downstream_losses_H{H}.npz"); LJ = {k: z[k] for k in MODELS_J}
        ts = pd.to_datetime(z["ts"]); yte = z["yte"]
        yt = y5.reindex(ts - pd.Timedelta(minutes=5 * H)).to_numpy(dtype="float32")
        assert np.isfinite(yt).all() and np.allclose((yt - yte) ** 2, LJ["persistence"], atol=1e-6), "y(t) misaligned"
        tg_in = z["inside"]; op_in = (yt >= SUPPORT[0]) & (yt <= SUPPORT[1])
        assert np.array_equal(tg_in, (yte >= SUPPORT[0]) & (yte <= SUPPORT[1]))
        inside = tg_in if split_name == "target" else op_in
        jsum.setdefault(f"H{H}", {"n_test": int(len(yte)), "agreement_pct": round(100 * float((tg_in == op_in).mean()), 1),
                                  "target_inside_n": int(tg_in.sum()), "last_input_inside_n": int(op_in.sum()),
                                  "both_inside_n": int((tg_in & op_in).sum())})
        dis = tg_in != op_in        # where the groupings disagree: last input and target on both sides of the support edge
        jsum[f"H{H}"].setdefault("disagreement", {
            "target_only_n": int((tg_in & ~op_in).sum()), "last_input_only_n": int((op_in & ~tg_in).sum()),
            "last_input_range": [float(yt[dis].min()), float(yt[dis].max())], "target_range": [float(yte[dis].min()), float(yte[dis].max())],
            "test_targets_48_52_pct": round(100 * float(((yte >= 48) & (yte <= 52)).mean()), 1),
            "rmse_target_inside_only": {k: round(rmse_m(LJ[k], tg_in & ~op_in), 2) for k in MODELS_J},
            "rmse_last_input_inside_only": {k: round(rmse_m(LJ[k], op_in & ~tg_in), 2) for k in MODELS_J}})
        jsum[f"H{H}"][split_name] = {"inside_n": int(inside.sum()), "outside_n": int((~inside).sum()),
                                     "rmse_inside": {k: round(rmse_m(LJ[k], inside), 3) for k in MODELS_J},
                                     "rmse_outside": {k: round(rmse_m(LJ[k], ~inside), 3) for k in MODELS_J}}
        pairs = [(v, "persistence") for v in VARIANTS] + [(v, "raw") for v in VARIANTS[1:]] + [("ae_repair", "rules_repair")]
        for a, b in pairs:                                                           # script 10 order
            for scope, sel in [("all", np.ones(len(yte), bool)), ("inside_support", inside)]:
                la, lb = LJ[a][sel], LJ[b][sel]; lo, hi = block_bootstrap_ci(la, lb, rng_j)
                jrows.append({"split": split_name, "H": H, "scope": scope, "a": a, "b": b, "n": int(sel.sum()),
                              "rmse_a": float(np.sqrt(la.mean())), "rmse_b": float(np.sqrt(lb.mean())),
                              "diff": float(np.sqrt(la.mean()) - np.sqrt(lb.mean())), "ci_lo": lo, "ci_hi": hi})
jdf = pd.DataFrame(jrows).round(4); jdf.to_csv(OUT_T / "downstream_support_operational.csv", index=False)
chk = jdf[jdf.split == "target"].merge(t12, on=["H", "scope", "a", "b"], suffixes=("", "_t12"))
assert len(chk) == len(t12) and all(np.allclose(chk[c], chk[c + "_t12"], atol=1e-4) for c in ["n", "rmse_a", "rmse_b", "diff", "ci_lo", "ci_hi"]), \
    "target split does not reproduce Table 12"
jrow = lambda s, H, a, b: jdf[(jdf.split == s) & (jdf.H == H) & (jdf.scope == "inside_support") & (jdf.a == a) & (jdf.b == b)].iloc[0]
for H in [1, 12, 72]:
    for s in ["target", "last_input"]:
        jsum[f"H{H}"][s]["differences_inside"] = {f"{a}-{b}": [float(r["diff"]), float(r["ci_lo"]), float(r["ci_hi"])] for a, b in
            [("rules_repair", "persistence"), ("ae_repair", "persistence"), ("union_repair", "persistence"), ("drop_faulty", "persistence"),
             ("raw", "persistence"), ("rules_repair", "raw"), ("ae_repair", "raw"), ("union_repair", "raw")] for r in [jrow(s, H, a, b)]}
for H in [1, 12, 72]:   # raw-data RMSE inside the support divided by repaired-data RMSE, under each split
    for s in ["target", "last_input"]:
        ri = jsum[f"H{H}"][s]["rmse_inside"]
        jsum[f"H{H}"][s]["raw_over_repair_inside"] = {v: round(ri["raw"] / ri[v], 2) for v in ["rules_repair", "ae_repair", "union_repair"]}
out["operational_support_split"] = {"support": list(SUPPORT), "table12_reproduced_by_target_split": True,
                                    "file": "results/tables/downstream_support_operational.csv", **jsum}
print(json.dumps(out["operational_support_split"], indent=1))

with open(OUT_T / "revision_checks.json", "w") as f: json.dump(out, f, indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "detection_per_window"}, indent=2)); print("done")
