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

if FAIL:
    print(f"\n{len(FAIL)} invariant(s) FAILED"); sys.exit(1)
print("\nAll release invariants pass.")
