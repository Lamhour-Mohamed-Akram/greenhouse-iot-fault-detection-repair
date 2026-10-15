# Script 22 - Tables for the supplementary material: window-length sensitivity of the detectors, block-length sensitivity
# of the temporal intervals, per-sensor imputation errors of all seven methods, and the inside-support errors under the
# target-based and the last-input grouping (from script 24, section j)
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
TAB = ROOT / "results" / "tables"; GEN = TAB
def f(x, d=2): return f"{x:.{d}f}"

# ---------------- detector window lengths
w = pd.read_csv(TAB / "detection_window_length_sensitivity.csv")
w["len"] = w.setting.str.extract(r"(\d+) min").astype(int)
w = w.sort_values(["detector", "len"])
DN = {"rules": "Rules (Hampel window)", "iforest": "Isolation Forest (rolling window)", "lstm_ae": "LSTM-AE (sequence length, retrained)"}
DEF = {"rules": 61, "iforest": 30, "lstm_ae": 60}
lines = [r"\begin{table}[!ht]", r"\centering", r"\footnotesize",
         r"\caption{Window-length sensitivity of the detectors on the 72 held-out scenarios (other settings at their defaults). Columns as in Table~\ref{tab:sensitivity}; default lengths in bold. The LSTM-AE row at 60~min comes from the retrained model of this analysis and differs slightly from the default model of Table~\ref{tab:detection}.}",
         r"\label{tab:windowlength}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llccccccccc}", r"\toprule",
         r"\textbf{Detector} & \textbf{Window} & \textbf{Macro-F1} & \textbf{Macro-F1 (5)} & \textbf{bias} & \textbf{drift} & \textbf{noise} & \textbf{spike} & \textbf{stuck} & \textbf{FP mean} & \textbf{FP max} \\", r"\midrule"]
for i, d in enumerate(["rules", "iforest", "lstm_ae"]):
    for _, r in w[w.detector == d].iterrows():
        L = f"{r.len} min"; L = r"\textbf{" + L + "}" if r.len == DEF[d] else L
        lines.append(f"{DN[d]} & {L} & {f(r.macro_f1, 3)} & {f(r.macro_f1_5, 3)} & {f(r.f1_bias, 3)} & {f(r.f1_drift, 3)} & {f(r.f1_noise, 3)} & {f(r.f1_spike, 3)} & {f(r.f1_stuck, 3)} & {100 * r.fp_mean:.2f}\\% & {100 * r.fp_max:.2f}\\% \\\\")
    if i < 2: lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(GEN / "tab_s_window_length.tex").write_text("\n".join(lines) + "\n")

# ---------------- block-length sensitivity of the temporal intervals
b = pd.read_csv(TAB / "downstream_block_sensitivity.csv")
VN = {"raw": "raw", "rules_repair": "rules$\\rightarrow$repair", "ae_repair": "LSTM-AE$\\rightarrow$repair", "union_repair": "union$\\rightarrow$repair",
      "drop_faulty": "drop-faulty", "persistence": "persistence"}
HN = {1: "5 min", 12: "1 h", 72: "6 h"}; SN = {"all": "full", "inside_support": "inside"}
# a limit within 0.01 of zero is shown with three decimals so that its sign is visible (the borderline limits cited in the text)
lim = lambda x: f"{x:+.3f}" if 0 < abs(x) < 0.01 else f"{x:+.2f}"
ci = lambda lo, hi: f"[{lim(lo)}, {lim(hi)}]"
lines = [r"\begin{longtable}{lllcccccc}",
         r"\caption{Sensitivity of the temporal intervals of Table~\ref{tab:inference} to the resampling scheme: pooled RMSE difference (first minus second) with 95\% moving-block bootstrap intervals for block lengths of 12, 24 and 48 samples, and with a bootstrap over contiguous runs of retained test samples ($n$ runs). The difference and the block-24 column are those of Table~\ref{tab:inference} (same bootstrap draw, which also gives the union rows not shown there); the block-12, block-48 and run-based columns are separate draws. Limits within 0.01 of zero are shown with three decimals.}\label{tab:blocks}\\",
         r"\toprule", r"\textbf{H} & \textbf{Set} & \textbf{Comparison} & $\Delta$ & \textbf{block 12} & \textbf{block 24} & \textbf{block 48} & \textbf{runs} & $n$ \\", r"\midrule", r"\endfirsthead",
         r"\multicolumn{9}{l}{\small\emph{Table~\thetable\ (continued)}} \\", r"\toprule", r"\textbf{H} & \textbf{Set} & \textbf{Comparison} & $\Delta$ & \textbf{block 12} & \textbf{block 24} & \textbf{block 48} & \textbf{runs} & $n$ \\", r"\midrule", r"\endhead",
         r"\bottomrule", r"\endfoot"]
prev = None
for _, r in b.iterrows():
    key = (r.H, r.scope)
    if prev is not None and key != prev: lines.append(r"\midrule")
    lead = f"{HN[r.H]} & {SN[r.scope]}" if key != prev else " & "
    lines.append(f"{lead} & {VN[r.a]} vs.\\ {VN[r.b]} & {r['diff']:+.2f} & {ci(r.b12_lo, r.b12_hi)} & {ci(r.b24_lo, r.b24_hi)} & {ci(r.b48_lo, r.b48_hi)} & {ci(r.seg_lo, r.seg_hi)} & {int(r.n_segments)} \\\\")
    prev = key
lines += [r"\end{longtable}"]
(GEN / "tab_s_blocks.tex").write_text("\n".join(lines) + "\n")
print("done")

# ---------------- per-sensor imputation RMSE, all seven methods
ps = pd.read_csv(TAB / "imputation_per_sensor_rmse.csv")
M = [("ffill", "forward-fill"), ("linear", "linear"), ("spline3", "spline"), ("knn", "KNN"), ("iterative", "iterative"), ("profile", "profile"), ("hybrid", "hybrid")]
SN = {"humidity": "Humidity", "temperature": "Temperature", "soil_moisture": "Soil-moisture index"}
G = [5, 15, 60, 240, 720]
lines = [r"\begin{table}[!ht]", r"\centering", r"\footnotesize",
         r"\caption{Per-sensor imputation RMSE in native units (mean over repeats and held-out windows) for all seven methods; bold = best per sensor, regime and gap length. Humidity in \% RH, temperature in \textdegree C, soil-moisture index in index units. The hybrid equals linear interpolation up to 1~h; in the node-outage regime the KNN and iterative imputers reduce to the column mean and coincide.}",
         r"\label{tab:imputation-sensor}", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{llccccc|ccccc}", r"\toprule",
         r" & & \multicolumn{5}{c|}{\textbf{Single-sensor gap}} & \multicolumn{5}{c}{\textbf{Node outage}} \\",
         r"\textbf{Sensor} & \textbf{Method} & 5 min & 15 min & 1 h & 4 h & 12 h & 5 min & 15 min & 1 h & 4 h & 12 h \\", r"\midrule"]
for k, s in enumerate(["humidity", "temperature", "soil_moisture"]):
    q = ps[ps.sensor == s]
    best = {(no, g): q[(q.node_outage == no) & (q.gap_min == g)][[c for c, _ in M]].round(2).min(axis=1).iloc[0] for no in (False, True) for g in G}
    for i, (c, name) in enumerate(M):
        cells = []
        for no in (False, True):
            for g in G:
                v = round(float(q[(q.node_outage == no) & (q.gap_min == g)][c].iloc[0]), 2)
                cells.append(r"\textbf{" + f"{v:.2f}" + "}" if v == best[(no, g)] else f"{v:.2f}")
        lines.append(f"{SN[s] if i == 0 else ''} & {name} & " + " & ".join(cells) + r" \\")
    if k < 2: lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(GEN / "tab_s_imputation_sensor.tex").write_text("\n".join(lines) + "\n")
print("S4 done")

# ---------------- inside-support errors: grouping by the true target vs. by the last observed input (script 24, section j)
op = pd.read_csv(TAB / "downstream_support_operational.csv"); opj = json.load(open(TAB / "revision_checks.json"))["operational_support_split"]
op = op[op.scope == "inside_support"]; GN = {"target": "target", "last_input": "last input"}
row = lambda s, H, a, b: op[(op.split == s) & (op.H == H) & (op.a == a) & (op.b == b)].iloc[0]
cell = lambda r: rf"\shortstack{{${r['diff']:+.2f}$\\{{\scriptsize $[{lim(r.ci_lo)}, {lim(r.ci_hi)}]$}}}}"
lines = [r"\begin{table}[!ht]", r"\centering", r"\small",
         r"\caption{Inside-support errors of the original level-predicting LSTMs with the test samples grouped by the true target (post hoc, as in Table~\ref{tab:inference}) or by the last observed value $y(t)$, which is known when the forecast is made; inside = value within the training support (0--50 index units). (a) Pooled RMSE in index units and the share of test samples on which the two groupings agree. (b) Pooled RMSE difference (first minus second) with the 95\% moving-block bootstrap interval (block 24, 2,000 replicates, the procedure of Table~\ref{tab:inference}; the target rows reproduce it). Values from script 24, section (j).}",
         r"\label{tab:opsplit}", r"\setlength{\tabcolsep}{4pt}",
         r"\textbf{(a)} RMSE inside the support\\[2pt]", r"\begin{tabular}{llcrccccc}", r"\toprule",
         r"\textbf{H} & \textbf{Grouping} & \textbf{Agree} & $n$ & \textbf{Persistence} & \textbf{Raw} & \textbf{Rules} & \textbf{LSTM-AE} & \textbf{Union} \\", r"\midrule"]
for H in [1, 12, 72]:
    for i, s in enumerate(["target", "last_input"]):
        r_ = {v: row(s, H, v, "persistence") for v in ["raw", "rules_repair", "ae_repair", "union_repair"]}
        agree = f"{opj[f'H{H}']['agreement_pct']:.1f}\\%" if i == 0 else ""
        lines.append(f"{HN[H] if i == 0 else ''} & {GN[s]} & {agree} & {int(r_['raw'].n)} & {r_['raw'].rmse_b:.2f} & " +
                     " & ".join(f"{r_[v].rmse_a:.2f}" for v in ["raw", "rules_repair", "ae_repair", "union_repair"]) + r" \\")
    if H != 72: lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", r"\\[8pt]", r"\textbf{(b)} Differences inside the support\\[2pt]", r"\begin{tabular}{llcccccc}", r"\toprule",
          r"\textbf{H} & \textbf{Grouping} & \shortstack{Rules\\$-$ pers.} & \shortstack{LSTM-AE\\$-$ pers.} & \shortstack{Union\\$-$ pers.} & \shortstack{Raw\\$-$ pers.} & \shortstack{Rules\\$-$ raw} & \shortstack{LSTM-AE\\$-$ raw} \\", r"\midrule"]
CMP = [("rules_repair", "persistence"), ("ae_repair", "persistence"), ("union_repair", "persistence"), ("raw", "persistence"), ("rules_repair", "raw"), ("ae_repair", "raw")]
for H in [1, 12, 72]:
    for i, s in enumerate(["target", "last_input"]):
        lines.append(f"{HN[H] if i == 0 else ''} & {GN[s]} & " + " & ".join(cell(row(s, H, a, b)) for a, b in CMP) + r" \\[2pt]")
    if H != 72: lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(GEN / "tab_s_opsplit.tex").write_text("\n".join(lines) + "\n")
print("S9 done")
