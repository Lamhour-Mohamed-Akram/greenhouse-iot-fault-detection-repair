"""System architecture figure from the deployment facts documented in Article 1."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mp

FIG = Path(__file__).resolve().parents[1] / "results" / "figures"
fig, ax = plt.subplots(figsize=(9.5, 3.1))
ax.axis("off")

def box(x, y, w, h, title, lines, fc="#eef3fb", ec="#2b5aa0", title_fs=8.5, fs=7):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                   fc=fc, ec=ec, lw=1.2))
    ax.text(x + w/2, y + h - 0.085, title, ha="center", va="top",
            fontsize=title_fs, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w/2, y + h - 0.19 - 0.088*i, ln, ha="center", va="top", fontsize=fs)

def arrow(x1, y1, x2, y2, label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.3))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.045, label, ha="center", fontsize=6.5, color="#444")

box(0.01, 0.30, 0.17, 0.62, "Greenhouse sensors",
    ["DHT22 (air T/RH)", "DS18B20 (soil T)", "Soil-moisture v1.2 (soil M)",
     "Digital LDR (light)", "CO$_2$ (model n/r)"])
box(0.235, 0.42, 0.135, 0.38, "IoT node", ["ESP32 (Wi-Fi)", "MQTT client"])
box(0.425, 0.42, 0.15, 0.38, "Gateway", ["Raspberry Pi 4", "MQTT broker,", "Python logger"])
box(0.63, 0.42, 0.145, 0.38, "Raw database",
    ["947,682 records", "18 Apr--22 May 2022", "median $\\Delta t \\approx$ 1 s"])
box(0.63, 0.005, 0.365, 0.30, "This work", [], fc="#eefaf0", ec="#2b7a3f")
ax.text(0.675, 0.115, "fault\ndetection", ha="center", fontsize=7)
ax.text(0.805, 0.115, "repair\n(imputation)", ha="center", fontsize=7)
ax.text(0.935, 0.115, "soil-moisture\nforecasting", ha="center", fontsize=7)
arrow(0.725, 0.09, 0.755, 0.09); arrow(0.86, 0.09, 0.885, 0.09)
arrow(0.18, 0.61, 0.235, 0.61, "wired/analog")
arrow(0.37, 0.61, 0.425, 0.61, "Wi-Fi / MQTT")
arrow(0.575, 0.61, 0.63, 0.61)
arrow(0.675, 0.42, 0.675, 0.31)
ax.text(0.615, 0.36, "1-min / 5-min\nresampling", ha="right", fontsize=6.5, color="#444")
fig.savefig(FIG / "fig12_architecture.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig12_architecture.pdf", bbox_inches="tight")
print("saved fig12")
