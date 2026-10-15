# Script 23 - Severity heat map of the detection benchmark redrawn from the saved per-severity F1 table (no grid over the cells)
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
ORIG_T = ROOT / "results" / "tables"; OUT_F = ROOT / "results" / "figures"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 10, "axes.grid": False})

sev = pd.read_csv(ORIG_T / "detection_benchmark_by_severity.csv").set_index(["detector", "fault_type"])[["low", "medium", "high"]]
NAMES = {"rules": "Rules", "iforest": "Isolation Forest", "lstm_ae": "LSTM-AE"}
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for ax, det in zip(axes, ["rules", "iforest", "lstm_ae"]):
    m = sev.loc[det]
    im = ax.imshow(m.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(3), ["low", "medium", "high"])
    ax.set_yticks(range(len(m)), m.index)
    ax.set_xticks([x - 0.5 for x in range(1, 3)], minor=True); ax.set_yticks([y - 0.5 for y in range(1, len(m))], minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    ax.set_title(NAMES[det], fontsize=10)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=9, weight="bold", color="white" if (m.values[i, j] < 0.15 or m.values[i, j] > 0.85) else "black")
fig.colorbar(im, ax=axes, shrink=0.85, label="F1")
fig.savefig(OUT_F / "fig6_detection_severity_heatmap.png"); fig.savefig(OUT_F / "fig6_detection_severity_heatmap.pdf"); plt.close(fig)
print("done")
