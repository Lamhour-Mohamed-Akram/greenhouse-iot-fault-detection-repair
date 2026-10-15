# Script 18 - Summary table and diagnostics for the random-mask control (outputs of script 17)
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
RC = ROOT / "results" / "random_mask_control"; GEN = RC
fits = pd.read_csv(RC / "per_fit_metrics.csv"); tin = pd.read_csv(RC / "temporal_inference.csv"); pdiff = pd.read_csv(RC / "paired_differences.csv")
masks = np.load(RC / "masks.npz"); diag = pd.read_csv(RC / "mask_diagnostics.csv")

# share of the out-of-range soil-moisture episode left unmasked by each random mask (diagnostic)
raw = pd.read_csv(ROOT / "data" / "data.csv.gz", encoding="latin-1")
raw.columns = ["Id", "humidity", "temperature", "soil_moisture", "soil_temp", "co2", "light", "date"]
raw["date"] = pd.to_datetime(raw["date"]); idx = pd.to_datetime(masks["train_index"])
sm = raw.sort_values("date").set_index("date")["soil_moisture"].resample("5min").median().reindex(idx).to_numpy(); neg = sm < 0
oor = [{"control": c, "mask_seed": ms, "oor_left_pct": round(100 * float((neg & ~masks[f"{p}_{ms}_soil_moisture"]).sum() / neg.sum()), 1)}
       for p, c in [("R", "random_mask_repair"), ("RB", "random_split_repair")] for ms in [101, 202, 303, 404, 505]]
oor = pd.DataFrame(oor); oor.to_csv(RC / "oor_left_unmasked.csv", index=False)
mech = []
t = np.load(RC / "test_set.npz"); yte, ins, pers = t["yte"], t["inside"], t["persistence"]
for f in sorted((RC / "predictions").glob("*.npz")):
    p = np.load(f)["pred"]; o = ~ins; tag = f.stem.rsplit("_seed", 1)[0]
    mech.append({"condition": tag, "seed": int(f.stem.split("seed")[1]), "slope_on_last_value_outside": float(np.polyfit(pers[o], p[o], 1)[0]),
                 "max_prediction": float(p.max())})
mech = pd.DataFrame(mech); mech.to_csv(RC / "extrapolation_diagnostics.csv", index=False)

def f(x, d=2): return f"{x:.{d}f}"
g = fits.groupby("group")
lines = [r"\begin{table*}[!t]", r"\footnotesize", r"\begin{center}",
         r"\caption{Random-mask control for the original level-predicting LSTM (5-min horizon, common screened test set). Random-mask$\rightarrow$repair masks, on the training partition only, the same number of observed values per channel as the rule detector, in runs of the same lengths, at random positions (5 mask seeds $\times$ 10 training seeds); genuine missing values and every other step are identical to rules$\rightarrow$repair. In the split variant the single 2,026-slot soil-moisture run is placed as two 1,013-slot runs. RMSE: mean $\pm$ sd over fits. $\Delta$ = pooled RMSE of the control minus that of rules$\rightarrow$repair (positive favors rules) with the 95\% bootstrap interval over contiguous runs of test samples; ``masks'' = number of the five mask seeds whose seed-averaged RMSE is above that of rules$\rightarrow$repair. OOR left = share of the out-of-range soil-moisture episode the control leaves unmasked in training.}",
         r"\label{tab:randomcontrol}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lccccccc}", r"\toprule",
         r"\textbf{Training data} & \textbf{RMSE full} & \textbf{RMSE inside} & \textbf{RMSE outside} & \textbf{$\Delta$ full [95\% CI]} & \textbf{$\Delta$ inside [95\% CI]} & \textbf{Masks full / inside} & \textbf{OOR left} \\", r"\midrule"]
for grp, name in [("persistence", "Persistence (reference)"), ("raw", "Raw (reference)"), ("rules_repair", "Rules$\\rightarrow$repair"),
                  ("random_mask_repair", "Random-mask$\\rightarrow$repair"), ("random_split_repair", "Random-mask$\\rightarrow$repair, split")]:
    q = g.get_group(grp)
    cell = lambda c: f(q[c].mean()) + ("" if len(q) == 1 else " $\\pm$ " + f(q[c].std()))
    if grp.startswith("random"):
        a = "random_all" if grp == "random_mask_repair" else "random_split_all"
        tf_ = tin[(tin.scope == "full") & (tin.a == a) & (tin.b == "rules_repair")].iloc[0]; ti_ = tin[(tin.scope == "inside_support") & (tin.a == a) & (tin.b == "rules_repair")].iloc[0]
        pf = pdiff[(pdiff.control == grp) & (pdiff.metric == "rmse")].iloc[0]; pi = pdiff[(pdiff.control == grp) & (pdiff.metric == "rmse_inside")].iloc[0]
        dfull = f"{tf_.delta:+.2f} [{tf_.run_lo:+.2f}, {tf_.run_hi:+.2f}]"; dins = f"{ti_.delta:+.2f} [{ti_.run_lo:+.2f}, {ti_.run_hi:+.2f}]"
        mk = f"{int(pf.masks_favoring_rules)} / {int(pi.masks_favoring_rules)}"
        ol = oor[oor.control == grp].oor_left_pct; olt = f"{ol.min():.0f}--{ol.max():.0f}\\%"
    else:
        dfull = dins = mk = "n/a"; olt = {"raw": "100\\%", "rules_repair": "0\\%", "persistence": "n/a"}[grp]
    lines.append(f"{name} & {cell('rmse')} & {cell('rmse_inside')} & {cell('rmse_outside')} & {dfull} & {dins} & {mk} & {olt} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\end{table*}"]
(GEN / "tab_random_control.tex").write_text("\n".join(lines) + "\n")
print(open(GEN / "tab_random_control.tex").read())
print(mech.assign(group=mech.condition.str.replace(r"_m\d+", "", regex=True)).groupby("group")[["slope_on_last_value_outside", "max_prediction"]].mean().round(2))
