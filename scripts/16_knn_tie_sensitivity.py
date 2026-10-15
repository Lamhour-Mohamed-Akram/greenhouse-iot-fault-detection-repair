# Script 16 - Sensitivity of the KNN imputer to distance ties
#
# The channels are quantized (integer soil-moisture index, 0.1-unit humidity and temperature), so many
# donors are exactly equidistant from a gap row and the choice among them depends on the sort order of
# the neighbour search, which can differ between numerical libraries and platforms. This script repeats
# the KNN part of the imputation benchmark (same gap positions as script 04) with ties broken at random:
# a jitter of 1e-6 times each channel's window standard deviation is added to all observed values, with
# five different seeds, and the equal-sensor normalized RMSE is compared with the unjittered run.
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
DATA = ROOT / "data" / "data.csv.gz"
OUT_T = TAB = ROOT / "results" / "tables"
SEED = 42; TARGETS = ["humidity", "temperature", "soil_moisture"]; COV = ["humidity", "temperature", "soil_moisture", "light"]
GAPS = [5, 15, 60, 240, 720]; REPEATS = 5

raw = pd.read_csv(DATA, encoding="latin-1"); raw.columns = ["Id", "humidity", "temperature", "soil_moisture", "soil_temp", "co2", "light", "date"]
raw["date"] = pd.to_datetime(raw["date"]); grid = raw.sort_values("date").set_index("date")[COV].resample("1min").median()
win_df = pd.read_csv(TAB / "clean_windows.csv", parse_dates=["start", "end"])
test_w = {}
for s, g in win_df.groupby("sensor"):
    st = sorted(g.start); n = max(1, min(2, len(st) - 1)); test_w[s] = st[n:]
sd_clean = json.load(open(OUT_T / "imputation_extended_headline.json"))["clean_sd"]

# script 04 draws the single-sensor regime first, then the node-outage regime, from one generator
def run_ordered(jitter_seed=None):
    rng = np.random.default_rng(SEED); out, ties, total = [], 0, 0
    for outage in [False, True]:
        for s in TARGETS:
            for t0 in test_w[s]:
                w0 = grid.loc[t0:t0 + pd.Timedelta(minutes=1439), COV].copy()
                if w0[s].isna().any(): w0[s] = w0[s].interpolate(limit_direction="both")
                for L in GAPS:
                    for r in range(REPEATS):
                        start = int(rng.integers(0, len(w0) - L)); w = w0.copy()
                        if outage: w.iloc[start:start + L, :] = np.nan
                        else: w.iloc[start:start + L, w.columns.get_loc(s)] = np.nan
                        X = w.values.astype(float); col = w.columns.get_loc(s)
                        if jitter_seed is not None:
                            jr = np.random.default_rng([jitter_seed, len(out)])
                            X = X + jr.normal(0, 1e-6, X.shape) * np.nanstd(w0.values, axis=0)
                        elif not outage:
                            donors = ~np.isnan(X[:, col])
                            D = np.sort(nan_euclidean_distances(X[start:start + L], X[donors]), axis=1)
                            ties += int(np.isclose(D[:, 9], D[:, 10]).sum()); total += L
                        est = KNNImputer(n_neighbors=10).fit_transform(X)[start:start + L, col]
                        truth = w0[s].iloc[start:start + L].values
                        out.append({"node_outage": outage, "sensor": s, "window_start": t0, "gap_min": L, "repeat": r,
                                    "nrmse": float(np.sqrt(np.mean((est - truth) ** 2))) / sd_clean[s]})
    return pd.DataFrame(out), ties, total

def eq(df):
    return df.groupby(["node_outage", "gap_min", "sensor"]).nrmse.mean().groupby(["node_outage", "gap_min"]).mean()
base, ties, total = run_ordered(None)
res = {"unjittered": eq(base)}
for js in range(5):
    d, _, _ = run_ordered(js); res[f"jitter seed {js}"] = eq(d)
tab = pd.DataFrame(res).round(3)
tab["jitter_min"] = tab[[c for c in tab if c.startswith("jitter")]].min(axis=1)
tab["jitter_max"] = tab[[c for c in tab if c.startswith("jitter")]].max(axis=1)
tab.to_csv(OUT_T / "imputation_knn_tie_sensitivity.csv")
print(tab.to_string())
print(f"single-sensor gap rows with a distance tie between the 10th and 11th donor: {ties} of {total} ({100 * ties / total:.0f}%)")
json.dump({"tie_rows": ties, "gap_rows": total, "tie_pct": round(100 * ties / total, 1)}, open(OUT_T / "imputation_knn_tie_headline.json", "w"), indent=2)
print("done")
