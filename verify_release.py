"""Release invariants for the greenhouse fault detection and repair study.

Checks the shipped data and authoritative result files against the invariants
reported in the manuscript. Exits nonzero if any invariant fails.
Run from the repository root: python verify_release.py
"""
import json
import sys

import pandas as pd

FAIL = []

def check(name, cond, detail=""):
    status = "ok " if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAIL.append(name)

raw = pd.read_csv("data/data.csv.gz", encoding="latin-1")
check("raw rows == 947,682", len(raw) == 947_682, f"({len(raw)})")

inj = pd.read_parquet("results/injected/injected_scenarios.parquet")
check("injection scenarios == 162", inj["scenario_id"].nunique() == 162)
check("injection rows == 233,280", len(inj) == 233_280, f"({len(inj)})")
check("1,440 samples per scenario", inj.groupby("scenario_id").size().eq(1440).all())

mf = pd.read_csv("results/injected/injection_manifest.csv")
check("manifest scenarios == 162", mf["scenario_id"].nunique() == 162)
check("9 screened-clean windows", mf.groupby(["sensor", "window_start"]).ngroups == 9)

det = pd.read_csv("results/tables/detection_benchmark_raw.csv")
check("held-out detector scenarios == 72", det["scenario_id"].nunique() == 72)

dn = pd.read_csv("results/tables/downstream_variants_raw.csv")
check("downstream seeds == 10 (42-51)",
      sorted(dn["seed"].unique().tolist()) == list(range(42, 52)))
check("five downstream variants", dn["variant"].nunique() == 5)
check("drop-faulty n_train == 88",
      dn.loc[dn.variant == "drop_faulty", "n_train"].eq(88).all())
check("n_test == 1,012", dn["n_test"].eq(1012).all())

head = json.load(open("results/tables/downstream_variants_headline.json"))
summ = pd.read_csv("results/tables/downstream_variants_summary.csv", header=[0, 1], index_col=0)
for v in ["raw", "rules_repair", "ae_repair", "union_repair", "drop_faulty"]:
    check(f"headline rmse[{v}] matches raw CSV",
          abs(head["rmse"][v] - dn.loc[dn.variant == v, "rmse"].mean()) < 5e-4)

import os
check("no superseded downstream files",
      not any(os.path.exists(f"results/tables/{f}") for f in
              ["downstream_impact_raw.csv", "downstream_impact_summary.csv",
               "downstream_headline.json"]))

for f in ["results/tables/fault_taxonomy_summary.csv",
          "results/tables/imputation_headline.json",
          "results/tables/detection_headline.json",
          "results/tables/headline_numbers.json"]:
    check(f"exists: {f}", os.path.exists(f))

# ---- extended analyses (scripts 07-12)
rv = pd.read_csv("results/tables/downstream_ref_raw.csv")
check("horizons == {1, 12, 72}", sorted(rv["H"].unique().tolist()) == [1, 12, 72])
check("seeds == 10 (42-51)", sorted(rv["seed"].dropna().unique().astype(int).tolist()) == list(range(42, 52)))
check("drop-faulty n_train (5 min) == 1,741",
      rv.loc[(rv.variant == "drop_faulty") & (rv.H == 1), "n_train"].eq(1741).all())
rh = json.load(open("results/tables/downstream_ref_headline.json"))
check("n_test (5 min) == 1,162", rh["test_sets"]["1"]["n_test"] == 1162)
check("row any-fault (raw-sample stuck) == 61.3%", abs(rh["flags"]["row_any_fault_pct_raw_stuck"] - 61.3) < 0.05)
bl = pd.read_csv("results/tables/downstream_baselines.csv")
check("persistence RMSE (5 min) == 1.29", abs(bl.loc[(bl.H == 1) & (bl.variant == "persistence"), "rmse"].iloc[0] - 1.29) < 0.005)
r2 = json.load(open("results/tables/detection_extended_headline.json"))
check("detection macro-F1 reproduced from saved models",
      r2["macro_f1"] == {"iforest": 0.333, "lstm_ae": 0.356, "rules": 0.444})
check("rules on real stream (raw-sample stuck) == Table 7 total (51.6%)",
      abs(r2["real_stream_coverage"]["rules_raw_stuck_pct"] - 51.6) < 0.05)
ab = pd.read_csv("results/tables/downstream_longgap_ablation_raw.csv")
check("ablation reports inside-support RMSE", "inside_rmse" in ab.columns)
eqs = pd.read_csv("results/tables/imputation_normalized_rmse_equal_sensor.csv")
check("equal-sensor imputation table has 14 rows", len(eqs) == 14)
bs = pd.read_csv("results/tables/downstream_block_sensitivity.csv")
check("block sensitivity covers 10 comparisons x 3 horizons x 2 scopes", len(bs) == 60)
check("policy-selection table exists", os.path.exists("results/tables/imputation_policy_selection_training_windows.csv"))
check("window-length sensitivity covers 9 settings", len(pd.read_csv("results/tables/detection_window_length_sensitivity.csv")) == 9)
ce = json.load(open("results/tables/characterization_extended_headline.json"))
check("variance collapse = 5 consecutive days", ce["variance_collapse"]["n_days"] == 5)
rc = json.load(open("results/random_mask_control/headline.json"))
check("random-mask control: 50 + 50 random fits, none failed", rc["n_fits_random"] == 50 and rc["n_fits_random_split"] == 50 and rc["n_failed"] == 0)
check("random-mask control reproduces script 10", rc["reproduction_max_abs_diff"] < 1e-6)
check("random-mask control: 120 prediction files", len(os.listdir("results/random_mask_control/predictions")) == 120)
cm_sel = pd.read_csv("results/change_models/selection.csv")
check("change models: rules->repair at 5 min selects ridge", cm_sel[(cm_sel.H == 1) & (cm_sel.variant == "rules_repair")].selected.iloc[0] == "ridge")
cm = pd.read_csv("results/change_models/test_metrics_all_candidates.csv")
check("change models: rules->repair ridge 5-min RMSE == 1.112", abs(cm[(cm.H == 1) & (cm.variant == "rules_repair") & (cm.candidate == "ridge")].rmse.iloc[0] - 1.112) < 0.001)
check("residual LSTM: 150 prediction files", len(os.listdir("results/residual_forecaster/predictions")) == 150)
check("change models: at 1 h rules->repair selects persistence and union->repair selects ridge", cm_sel[(cm_sel.H == 12) & (cm_sel.variant == "rules_repair")].selected.iloc[0] == "persistence" and cm_sel[(cm_sel.H == 12) & (cm_sel.variant == "union_repair")].selected.iloc[0] == "ridge")
check("change models: 210 prediction files", len(os.listdir("results/change_models/predictions")) == 210)
kt = json.load(open("results/tables/imputation_knn_tie_headline.json"))
check("KNN tie analysis present", kt["tie_pct"] > 0)
ro = pd.read_csv("results/tables/downstream_rolling_origin_summary.csv")
check("rolling-origin blocks == 3", ro["block"].nunique() == 3)
# ---- second revision round (scripts 01, 07, 09, 14, 22, 24)
hn = json.load(open("results/tables/headline_numbers.json"))
check("dropout reported on the grid basis (26.4; event 26.34, grid 26.36)",
      hn["dropout_pct_reported"] == 26.4 and hn["dropout_pct_event_basis"] == 26.34 and hn["dropout_pct_grid_slot_basis"] == 26.36 and "dropout_pct_of_span" not in hn)
check("longest in-range stuck episode 5.5 h (5 May); 153.1-h run is the out-of-range episode",
      hn["stuck_longest_hours"] == 5.5 and hn["stuck_longest_start"].startswith("2022-05-05") and hn["longest_constant_run_hours"] == 153.1
      and hn["longest_constant_run_is_out_of_range"] and hn["longest_constant_run_value"] == -85.0)
tx = pd.read_csv("results/tables/fault_taxonomy_summary.csv").set_index("sensor")
check("Table 7 soil-moisture total == 51.59", tx.loc["soil_moisture", "total_faulty_pct"] == 51.59)
bt = pd.read_csv("results/tables/bounds_table.csv").set_index("sensor")
check("bounds table: soil-moisture index in index units, 0-100 = nominal range of the mapping",
      bt.loc["soil_moisture", "unit"].startswith("index units") and "nominal range of the firmware mapping" in bt.loc["soil_moisture", "source_bounds"]
      and "impossible" not in bt.loc["soil_moisture", "source_bounds"])
check("superseded script-09 policy table renamed",
      not os.path.exists("results/tables/imputation_policy_training_period_only.csv") and os.path.exists("results/tables/imputation_policy_pooled_superseded.csv"))
bs24 = bs.merge(pd.read_csv("results/tables/downstream_temporal_inference.csv"), on=["H", "scope", "a", "b"], suffixes=("", "_t12"))
check("S8 block-24 column and difference == Table 12 (60 rows, same draw)",
      len(bs24) == 60 and (bs24.b24_lo == bs24.ci_lo).all() and (bs24.b24_hi == bs24.ci_hi).all() and (bs24["diff"] == bs24["diff_t12"]).all())
s8 = open("results/tables/tab_s_blocks.tex").read()
check("S8 caption corrected and borderline limits shown with 3 decimals",
      "can differ from Table" not in s8 and "[-0.35, +0.005]" in s8 and "[-0.005, +0.51]" in s8)
rc2 = json.load(open("results/tables/revision_checks.json"))
pt = rc2["persistence_test_sets"]
check("persistence RMSE 1.29 on the submitted test set (n = 1,012, grid stuck rule)",
      pt["submitted_grid_stuck_rule"]["n_test"] == 1012 and abs(pt["submitted_grid_stuck_rule"]["persistence_rmse"] - 1.29) < 0.005)
check("persistence RMSE 1.29 on the revised test set (n = 1,162, raw stuck rule)",
      pt["revised_raw_stuck_rule"]["n_test"] == 1162 and abs(pt["revised_raw_stuck_rule"]["persistence_rmse"] - 1.29) < 0.005)
f1m = rc2["flag_shares_soil_moisture"]["one_min_slots_whole_deployment"]; f5m = rc2["flag_shares_soil_moisture"]["five_min_training_rows"]
check("1-min stuck shares: median rule 45.3 / 26.7, raw rule 21.5 / 2.9 (incl. / excl. out-of-range)",
      (f1m["stuck_median_rule_incl_oor_pct"], f1m["stuck_median_rule_excl_oor_pct"], f1m["stuck_raw_rule_incl_oor_pct"], f1m["stuck_raw_rule_excl_oor_pct"]) == (45.3, 26.7, 21.5, 2.9))
check("5-min training rows: total rule flags 89.6 vs 52.4, stuck only excl. out-of-range 40.6 vs 3.3",
      (f5m["total_rule_flag_median_rule_pct"], f5m["total_rule_flag_raw_rule_pct"], f5m["stuck_only_excl_oor_median_rule_pct"], f5m["stuck_only_excl_oor_raw_rule_pct"]) == (89.6, 52.4, 40.6, 3.3))
check("rules on the real stream flag the same slots as the Table 7 total", f1m["table7_comparison"]["identical_slot_sets"])
rm4 = rc2["rq4_repair_mapping_rules_mask"]["per_channel"]["soil_moisture"]
check("RQ4 profile-filled target rows: 139 in 4-12 h runs + 3,170 in > 12 h runs = 3,309",
      rm4["rows_runs_4_to_12h"] == 139 and rm4["rows_runs_gt_12h"] == 3170
      and rm4["rows_runs_4_to_12h"] + rm4["rows_runs_gt_12h"] == rh["long_gap"]["train_target_rows_profile_filled_gt4h"]["rules"])
co = rc2["co2"]
check("CO2: 1-min median 113.3 ppm, 70.4% below 150; R2 = 97.8% of raw samples",
      co["time_basis_1min"]["median"] == 113.3 and co["time_basis_1min"]["pct_below_150"] == 70.4 and co["per_regime"]["R2"]["raw_sample_share_pct"] == 97.8)
check("CO2 per-regime medians 122.6 / 101.6 / 223.8 and script 07 agrees",
      [co["per_regime"][r]["median_1min"] for r in ["R1", "R2", "R3"]] == [122.6, 101.6, 223.8]
      and ce["co2"]["time_basis_1min"]["median"] == co["time_basis_1min"]["median"]
      and all(ce["co2"]["per_regime"][r]["median_night_22_05_1min"] == co["per_regime"][r]["median_night_22_05_1min"] for r in ["R1", "R2", "R3"]))
pw = pd.read_csv("results/tables/detection_paired_differences_by_window.csv")
check("per-window detection differences: 18 comparisons x 4 windows, 35 distinct bootstrap resamples",
      len(pw) == 18 and pw.n_windows.eq(4).all() and pw.distinct_bootstrap_resamples.eq(35).all()
      and ((pw.n_a_better + pw.n_b_better + pw.n_tied) == 4).all())

# ---- third revision round (script 24, sections g-i)
ep = rc2["soil_moisture_oor_episode"]
check("soil-moisture OOR episode 18 Apr 14:38 to 26 Apr 12:11; constant -85 from 20 Apr 02:41 to 26 Apr 11:48",
      ep["first_oor_sample"].startswith("2022-04-18 14:38") and ep["last_oor_sample"].startswith("2022-04-26 12:11")
      and ep["constant_minus85_run"] == ["2022-04-20 02:41:06", "2022-04-26 11:48:54"])
ccs = rc2["clean_screen_constancy_criterion"]
check("clean screen, soil moisture: 24.4% / 3 windows (median criterion) vs 48.1% / 9 windows, 6 before the boundary (raw-sample criterion)",
      (ccs["soil_moisture"]["median_criterion"]["clean_minutes_pct"], ccs["soil_moisture"]["median_criterion"]["n_24h_windows"],
       ccs["soil_moisture"]["raw_sample_criterion"]["clean_minutes_pct"], ccs["soil_moisture"]["raw_sample_criterion"]["n_24h_windows"],
       len(ccs["soil_moisture"]["raw_sample_criterion"]["windows_before_boundary"])) == (24.4, 3, 48.1, 9, 6))
check("clean screen: humidity 5 windows and temperature 2 windows (0 before the boundary) under both criteria",
      all(ccs["humidity"][c]["n_24h_windows"] == 5 for c in ["median_criterion", "raw_sample_criterion"])
      and all(ccs["temperature"][c]["n_24h_windows"] == 2 and not ccs["temperature"][c]["windows_before_boundary"] for c in ["median_criterion", "raw_sample_criterion"]))
check("no raw-stuck minute inside a window selected by the median-based screen",
      all(ccs[s]["raw_stuck_minutes_inside_median_screen_windows"] == 0 for s in ccs))
sp = pd.read_csv("results/tables/imputation_policy_soil_raw_screen.csv")
check("soil-moisture windows of the raw-sample screen select linear in every row (differs from Table 10 only at 12-h single-sensor gaps)",
      sp.selected.eq("linear").all() and sp.differs_from_table10.sum() == 1)
ops = rc2["operational_support_split"]; opd = pd.read_csv("results/tables/downstream_support_operational.csv")
t12 = pd.read_csv("results/tables/downstream_temporal_inference.csv")
m12 = opd[opd.split == "target"].merge(t12, on=["H", "scope", "a", "b"], suffixes=("", "_t12"))
check("operational split: the target grouping reproduces Table 12 (all pairs, intervals included)",
      len(m12) == len(t12) and all((m12[c] - m12[c + "_t12"]).abs().max() < 1e-4 for c in ["rmse_a", "rmse_b", "diff", "ci_lo", "ci_hi"]))
check("operational split: groupings agree on 82.1% / 77.8% / 72.7%; last-input inside n = 510 / 498 / 439",
      [ops[h]["agreement_pct"] for h in ["H1", "H12", "H72"]] == [82.1, 77.8, 72.7]
      and [ops[h]["last_input"]["inside_n"] for h in ["H1", "H12", "H72"]] == [510, 498, 439])
o1 = ops["H1"]["last_input"]
check("operational split, 5 min, inside RMSE: persistence 1.350, rules 1.560, AE 1.704, union 1.862, raw 5.717",
      [o1["rmse_inside"][k] for k in ["persistence", "rules_repair", "ae_repair", "union_repair", "raw"]] == [1.35, 1.56, 1.704, 1.862, 5.717])
dd = o1["differences_inside"]
check("operational split, 5 min: rules - persistence +0.21 (interval includes zero), AE +0.35 and union +0.51 (intervals exclude zero)",
      round(dd["rules_repair-persistence"][0], 2) == 0.21 and dd["rules_repair-persistence"][1] < 0 < dd["rules_repair-persistence"][2]
      and round(dd["ae_repair-persistence"][0], 2) == 0.35 and dd["ae_repair-persistence"][1] > 0
      and round(dd["union_repair-persistence"][0], 2) == 0.51 and dd["union_repair-persistence"][1] > 0)
check("operational split, 5 min: rules - raw -4.16, AE - raw -4.01 (intervals exclude zero)",
      round(dd["rules_repair-raw"][0], 2) == -4.16 and round(dd["ae_repair-raw"][0], 2) == -4.01
      and dd["rules_repair-raw"][2] < 0 and dd["ae_repair-raw"][2] < 0)
check("operational split, 1 h and 6 h: every repaired variant worse than persistence (rules +0.97 / +2.51); repair - raw -3.6 to -4.5",
      all(ops[h]["last_input"]["differences_inside"][f"{v}-persistence"][1] > 0 for h in ["H12", "H72"] for v in ["rules_repair", "ae_repair", "union_repair"])
      and round(ops["H12"]["last_input"]["differences_inside"]["rules_repair-persistence"][0], 2) == 0.97
      and round(ops["H72"]["last_input"]["differences_inside"]["rules_repair-persistence"][0], 2) == 2.51
      and all(-4.5 < ops[h][s]["differences_inside"][f"{v}-raw"][0] < -3.6 for h in ["H12", "H72"] for s in ["target", "last_input"] for v in ["rules_repair", "ae_repair", "union_repair"]))
rt1, rl1 = ops["H1"]["target"]["raw_over_repair_inside"].values(), ops["H1"]["last_input"]["raw_over_repair_inside"].values()
check("raw/repaired pooled inside-support RMSE ratio at 5 min: 3.9-4.9 by target, 3.1-3.7 by last input (abstract)",
      (round(min(rt1), 1), round(max(rt1), 1), round(min(rl1), 1), round(max(rl1), 1)) == (3.9, 4.9, 3.1, 3.7))
dp = pd.read_csv("results/tables/detection_paired_differences.csv").set_index(["fault_type", "a", "b"]).loc[("bias", "rules", "lstm_ae")]
check("LSTM-AE vs rules on bias: 0.16, interval 0.002 to 0.32 (Section 4.2)", (round(-dp.mean_diff, 2), round(-dp.ci_hi, 3), round(-dp.ci_lo, 2)) == (0.16, 0.002, 0.32))
check("raw/repaired inside-support RMSE ratio at 5 min spans 3.1-4.9 over both groupings",
      round(min(min(ops["H1"][s]["raw_over_repair_inside"].values()) for s in ["target", "last_input"]), 1) == 3.1
      and round(max(max(ops["H1"][s]["raw_over_repair_inside"].values()) for s in ["target", "last_input"]), 1) == 4.9)

for f in ["results/models/lstm_ae_humidity.keras", "results/models/lstm_ae_temperature.keras", "results/models/lstm_ae_soil_moisture.keras",
          "results/tables/regimes.csv", "results/tables/bounds_table.csv", "results/tables/detection_threshold_sensitivity.csv",
          "results/tables/imputation_normalized_rmse.csv", "results/tables/downstream_longgap_ablation.csv"]:
    check(f"exists: {f}", os.path.exists(f))

if FAIL:
    print(f"\n{len(FAIL)} invariant(s) FAILED"); sys.exit(1)
print("\nAll release invariants pass.")
