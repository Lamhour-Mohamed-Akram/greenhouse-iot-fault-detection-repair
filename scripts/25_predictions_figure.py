# Script 25 - Manuscript Figure 8 redrawn from saved predictions, no training: 5-min-ahead predictions of the level LSTMs
# trained on raw and on rules-repaired data (seed 42), persistence and the training support. The predictions are those
# saved by script 17, which reproduces the same fits of script 10 (results/random_mask_control/reproduction_check.csv).
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
RMC = ROOT / "results" / "random_mask_control"; OUT_F = ROOT / "results" / "figures"
SUPPORT = (0.0, 50.0)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})  # script 10
t = np.load(RMC / "test_set.npz"); ts = pd.to_datetime(t["ts"]); yt = t["yte"]
pred = {v: np.load(RMC / "predictions" / f"{v}_seed42.npz") for v in ["raw", "rules_repair"]}
for z in pred.values(): assert np.array_equal(z["ts"], t["ts"]) and np.array_equal(z["yte"], yt)

def broken(ts, vals):          # script 10: the line is broken at every gap in the retained samples
    t_ = pd.DatetimeIndex(ts); v = np.asarray(vals, float).copy()
    out_t, out_v = [t_[0]], [v[0]]
    for k in range(1, len(t_)):
        if (t_[k] - t_[k - 1]) > pd.Timedelta(minutes=5):
            out_t.append(t_[k - 1] + pd.Timedelta(minutes=5)); out_v.append(np.nan)
        out_t.append(t_[k]); out_v.append(v[k])
    return pd.DatetimeIndex(out_t), np.array(out_v)

fig, ax = plt.subplots(figsize=(10, 3.2))
bt, bv = broken(ts, yt); ax.plot(bt, bv, color="black", lw=1.2, label="observed (gaps broken)")
pt, pv = broken(ts, pred["raw"]["pred"]); ax.plot(pt, pv, color="tab:red", lw=0.8, alpha=0.8, label="raw-trained")
pt, pv = broken(ts, pred["rules_repair"]["pred"]); ax.plot(pt, pv, color="tab:olive", lw=0.8, alpha=0.9, label="rules $\\rightarrow$ repair")
pt, pv = broken(ts, t["persistence"]); ax.plot(pt, pv, color="tab:gray", lw=0.8, ls="--", alpha=0.8, label="persistence")
ax.axhspan(SUPPORT[0], SUPPORT[1], color="tab:blue", alpha=0.07, label="training support")
ax.set_ylabel("Soil-moisture index (index units)"); ax.legend(fontsize=8, ncol=5, loc="upper right")
ax.set_title("5-min-ahead predictions on the common untouched test set (seed 42)", fontsize=9)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(OUT_F / "fig11_downstream_predictions.png"); fig.savefig(OUT_F / "fig11_downstream_predictions.pdf"); plt.close(fig)
inside = (yt >= SUPPORT[0]) & (yt <= SUPPORT[1]); p = pred["rules_repair"]["pred"]; r = pred["raw"]["pred"]
print("seed 42 inside RMSE rules", round(float(np.sqrt(((p - yt)[inside] ** 2).mean())), 2), "raw", round(float(np.sqrt(((r - yt)[inside] ** 2).mean())), 2),
      "raw mean error inside", round(float((r - yt)[inside].mean()), 2), "rules max prediction", round(float(p.max()), 1))
