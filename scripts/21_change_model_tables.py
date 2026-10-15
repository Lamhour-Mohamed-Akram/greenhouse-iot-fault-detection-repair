# Script 21 - Tables for the level-free change models and the residual LSTM (outputs of scripts 19 and 20)
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
CM = ROOT / "results" / "change_models"; RF = ROOT / "results" / "residual_forecaster"; GEN = CM
VN = {"raw": "Raw", "rules_repair": "Rules$\\rightarrow$repair", "ae_repair": "LSTM-AE$\\rightarrow$repair", "union_repair": "Union$\\rightarrow$repair", "drop_faulty": "Drop-faulty"}
CN = {"persistence": "persistence", "ridge": "ridge", "lstm": "LSTM"}
HN = {1: "5 min", 12: "1 h", 72: "6 h"}
def f(x, d=2): return f"{x:.{d}f}"

# ---------------- level-free change models (selected candidate per variant and horizon)
sel = pd.read_csv(CM / "selection.csv"); tm = pd.read_csv(CM / "test_metrics_all_candidates.csv"); inf = pd.read_csv(CM / "temporal_inference.csv")
_dm = inf[(inf.H == 72) & (inf.scope == "full") & (inf.a == "drop_faulty|selected") & (inf.b == "persistence")].iloc[0]
lines = [r"\begin{table*}[!t]", r"\footnotesize", r"\begin{center}",
         r"\caption{Level-free change models (prediction = last observed value + predicted change; no absolute soil-moisture level among the inputs). For each training variant and horizon the candidate (persistence, ridge regression, level-free LSTM) with the lowest error on a chronological validation slice at the end of the training period is selected (detection, normalization and repair for that slice use only the earlier part of the training period) and then evaluated on the common test set. Ridge = RMSE of the ridge candidate for every variant (same model family, no selection). RMSE on the full test set and inside/outside the training support; SS = skill relative to persistence; $\Delta$ = pooled RMSE difference to persistence with the 95\% bootstrap interval over contiguous runs of test samples (n/a when persistence is selected). For these exploratory models we report this more conservative bootstrap rather than the moving-block bootstrap of Table~\ref{tab:inference}; with the moving-block bootstrap the 6-h drop-faulty difference to persistence would exclude zero (" + f"${_dm.b24_lo:.3f}$ to ${_dm.b24_hi:.3f}$" + r"). Bold row label: rules$\rightarrow$repair, the primary repaired variant; bold does not mark the best value. The selection among candidates is part of an exploratory analysis added after the main results; the intervals do not account for it. Persistence: " + ", ".join(f(tm[(tm.H == h) & (tm.candidate == "persistence")].rmse.iloc[0]) for h in [1, 12]) + " and " + f(tm[(tm.H == 72) & (tm.candidate == "persistence")].rmse.iloc[0]) + r" at 5~min, 1~h and 6~h.}",
         r"\label{tab:changemodels}", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{llccccccc}", r"\toprule",
         r"\textbf{Horizon} & \textbf{Training data} & \textbf{Selected} & \textbf{RMSE full} & \textbf{SS} & \textbf{RMSE inside} & \textbf{RMSE outside} & \textbf{Ridge} & \textbf{$\Delta$ vs.\ persistence [95\% CI]} \\", r"\midrule"]
for H in [1, 12, 72]:
    pr = tm[(tm.H == H) & (tm.candidate == "persistence")].rmse.iloc[0]; first = True
    for v in ["raw", "rules_repair", "ae_repair", "union_repair", "drop_faulty"]:
        c = sel[(sel.H == H) & (sel.variant == v)].selected.iloc[0]
        q = tm[(tm.H == H) & (tm.variant == v) & (tm.candidate == c)]
        rm, ri, ro = q.rmse.mean(), q.rmse_inside.mean(), q.rmse_outside.mean()
        r = inf[(inf.H == H) & (inf.scope == "full") & (inf.a == f"{v}|selected") & (inf.b == "persistence")].iloc[0]
        d = "n/a" if c == "persistence" else f"{r['diff']:+.2f} [{r.run_lo:+.2f}, {r.run_hi:+.2f}]"
        bold = v == "rules_repair"
        name = VN[v]; rms = f(rm)
        if bold: name = r"\textbf{" + name + "}"   # bold only on the row label
        rg = tm[(tm.H == H) & (tm.variant == v) & (tm.candidate == "ridge")].rmse.mean()
        lines.append(f"{HN[H] if first else ''} & {name} & {CN[c]} & {rms} & {f(1 - rm / pr)} & {f(ri)} & {f(ro)} & {f(rg)} & {d} \\\\"); first = False
    if H != 72: lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\end{table*}"]
(GEN / "tab_change_models.tex").write_text("\n".join(lines) + "\n")

# ---------------- residual LSTM (supplementary)
rs = pd.read_csv(RF / "per_fit_metrics.csv"); g = rs.groupby(["H", "variant"])
lines = [r"\noindent\begin{minipage}{\textwidth}", r"\begin{center}", r"\footnotesize",
         r"\captionof{table}{Residual LSTM (LSTM of Table~\ref{tab:downstream} trained on the change $y(t+H)-y(t)$, prediction = last observed value + predicted change, absolute level still among the inputs). RMSE mean $\pm$ sd over 10 seeds on the common test set.}",
         r"\label{tab:residual}", r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{lccc|ccc}", r"\toprule",
         r" & \multicolumn{3}{c|}{\textbf{RMSE full}} & \multicolumn{3}{c}{\textbf{RMSE inside support}} \\",
         r"\textbf{Training data} & 5 min & 1 h & 6 h & 5 min & 1 h & 6 h \\", r"\midrule"]
TB = ROOT / "results" / "tables"
bl = pd.read_csv(TB / "downstream_baselines.csv"); sd = pd.read_csv(TB / "downstream_support_decomposition.csv")
pers = {h: (bl[(bl.H == h) & (bl.variant == "persistence")].rmse.iloc[0], sd[(sd.H == h) & (sd.variant == "persistence")].inside_rmse.iloc[0]) for h in [1, 12, 72]}
lines.append("Persistence & " + " & ".join(f(pers[h][0]) for h in [1, 12, 72]) + " & " + " & ".join(f(pers[h][1]) for h in [1, 12, 72]) + r" \\")
for v in ["raw", "rules_repair", "ae_repair", "union_repair", "drop_faulty"]:
    full = [f(g.get_group((h, v)).rmse.mean()) + " $\\pm$ " + f(g.get_group((h, v)).rmse.std()) for h in [1, 12, 72]]
    ins = [f(g.get_group((h, v)).rmse_inside.mean()) for h in [1, 12, 72]]
    lines.append(f"{VN[v]} & " + " & ".join(full) + " & " + " & ".join(ins) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\end{minipage}", r"\vspace{10pt}"]
(GEN / "tab_residual.tex").write_text("\n".join(lines) + "\n")
print(open(GEN / "tab_change_models.tex").read()); print("done")
