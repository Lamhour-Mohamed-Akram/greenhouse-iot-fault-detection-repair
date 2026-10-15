# Script 11 - Deployment overview and real-stream detection figures
#
# Redraws the overview figure with readable time axes and the real-stream detection figure with
# the raw-sample stuck rule (detectors as in script 03, autoencoder loaded from results/models).
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
DATA = ROOT / "data" / "data.csv.gz"
RES = ROOT / "results"; OUT_T = RES / "tables"; OUT_F = RES / "figures"
TAB, INJ, MODELS, OFIG, ORIG_T, MS_F = OUT_T, RES / "injected", RES / "models", OUT_F, OUT_T, OUT_F
OUT_T.mkdir(parents=True, exist_ok=True); OUT_F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
SENSORS = ["humidity", "temperature", "soil_moisture", "soil_temp", "co2", "light"]
LABELS = {"humidity": "Air humidity (%)", "temperature": "Air temperature (degC)", "soil_moisture": "Soil-moisture index",
          "soil_temp": "Soil temperature (degC)", "co2": "CO2 (ppm)", "light": "Light (binary)"}
VALID_RANGE = {"humidity": (0, 100), "temperature": (-5, 55), "soil_moisture": (0, 100), "soil_temp": (-5, 55), "co2": (0, 5000), "light": (0, 1)}

raw = pd.read_csv(DATA, encoding="latin-1"); raw.columns = ["Id"] + SENSORS + ["date"]
raw["date"] = pd.to_datetime(raw["date"]); raw = raw.sort_values("date").reset_index(drop=True)
df1 = raw.set_index("date")[SENSORS].resample("1min").median()
dt = raw["date"].diff().dt.total_seconds(); gi = np.where(dt > 300)[0]
gaps = pd.DataFrame({"gap_start": raw["date"].iloc[gi - 1].values, "gap_end": raw["date"].iloc[gi].values})

def raw_stuck_1min(s):
    x = raw[s]; d = raw["date"]; gap = d.diff().dt.total_seconds() > 300
    g = ((x.diff() != 0) | gap).cumsum(); st = d.groupby(g).transform("first"); en = d.groupby(g).transform("last")
    m = pd.Series(((en - st).dt.total_seconds() > 3600).values, index=d)
    return m.groupby(pd.Grouper(freq="1min")).any().reindex(df1.index, fill_value=False)
def hampel(x, k=6, win=61):
    med = x.rolling(win, center=True, min_periods=10).median(); mad = (x - med).abs().rolling(win, center=True, min_periods=10).median()
    return ((x - med).abs() > k * 1.4826 * mad) & mad.gt(0) & x.notna()

# ---------------- Fig 1: deployment overview with readable time axis
fig, axes = plt.subplots(len(SENSORS), 1, figsize=(10, 11), sharex=True)
for ax, s in zip(axes, SENSORS):
    ax.plot(df1.index, df1[s], lw=0.4, color="tab:blue")
    lo, hi = VALID_RANGE[s]; bad = df1[s].where((df1[s] < lo) | (df1[s] > hi))
    ax.plot(df1.index, bad, ".", ms=2, color="tab:red", label="out-of-range")
    if s not in ("light", "soil_temp"):
        sp = hampel(df1[s])
        if sp.any(): ax.plot(df1.index, df1[s].where(sp), ".", ms=3, color="tab:orange", label="spike")
    if s != "light":
        stk = raw_stuck_1min(s) & ~((df1[s] < lo) | (df1[s] > hi))
        if stk.any(): ax.plot(df1.index, df1[s].where(stk), ".", ms=1, color="tab:purple", label="stuck-at")
    for _, g in gaps.iterrows(): ax.axvspan(g.gap_start, g.gap_end, color="gray", alpha=0.35)
    ax.set_ylabel(LABELS[s], fontsize=9)
    if ax.get_legend_handles_labels()[0]: ax.legend(loc="upper right", fontsize=9, markerscale=4)
axes[0].set_title("Raw greenhouse IoT deployment with real fault overlays (gray = dropout gaps)")
axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3)); axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
fig.savefig(OUT_F / "fig1_deployment_overview_faults.png"); fig.savefig(OUT_F / "fig1_deployment_overview_faults.pdf"); plt.close(fig)

# ---------------- Fig 7: detectors on the real soil-moisture stream (rules with raw-sample stuck)
sm = df1["soil_moisture"]
lo, hi = VALID_RANGE["soil_moisture"]
flags_rules = (((sm < lo) | (sm > hi)) & sm.notna()) | (raw_stuck_1min("soil_moisture") & sm.notna()) | hampel(sm) | sm.isna()
# learned detectors exactly as in script 03 (clean training windows, saved AE)
data = pd.read_parquet(INJ / "injected_scenarios.parquet"); mf = pd.read_csv(INJ / "injection_manifest.csv", parse_dates=["window_start"])
win_df = pd.read_csv(TAB / "clean_windows.csv", parse_dates=["start", "end"])
g = win_df[win_df.sensor == "soil_moisture"]; starts = sorted(g.start); train_w = starts[:max(1, min(2, len(starts) - 1))]
parts = []
for t0 in train_w:
    sc = mf[(mf.sensor == "soil_moisture") & (mf.window_start == t0)].scenario_id.iloc[0]; d = data[data.scenario_id == sc]
    parts.append(pd.Series(d.value_clean.values, index=pd.DatetimeIndex(d.timestamp)))
ct = pd.concat(parts); mu, sd = ct.mean(), ct.std()
prof = ct.groupby(ct.index.hour * 60 + ct.index.minute).mean().reindex(range(1440)).interpolate(limit_direction="both").rolling(31, center=True, min_periods=1).mean()
def features(x):
    minute = x.index.hour * 60 + x.index.minute; pr = prof.reindex(minute).values
    grp = (x.diff() != 0).cumsum(); runlen = x.groupby(grp).cumcount() + 1
    return pd.DataFrame({"z": (x - mu) / sd, "diff1": x.diff(), "roll_std": x.rolling(30, min_periods=5).std(), "profile_dev": x.values - pr, "runlen": runlen}, index=x.index).ffill().bfill()
iforest = IsolationForest(n_estimators=200, contamination=0.01, random_state=42).fit(features(ct))
flags_if = (iforest.predict(features(sm)) == -1) | sm.isna().values
ae = tf.keras.models.load_model(MODELS / "lstm_ae_soil_moisture.keras", compile=False)
def pte(z, n, seq=60):
    v = z.values.astype("float32"); idx = np.arange(seq, len(v) + 1); X = np.stack([v[i - seq:i] for i in idx])[..., None]
    se = (ae.predict(X, verbose=0) - X) ** 2; es, c = np.zeros(n), np.zeros(n)
    for k, end in enumerate(idx - 1): es[end - seq + 1:end + 1] += se[k, :, 0]; c[end - seq + 1:end + 1] += 1
    return es / np.maximum(c, 1)
thr = float(np.quantile(pte((ct - mu) / sd, len(ct)), 0.995))
flags_ae = (pte(((sm - mu) / sd).ffill().bfill(), len(sm)) > thr) | sm.isna().values
print("real-stream flagged %:", {"rules": round(100 * flags_rules.mean(), 1), "iforest": round(100 * flags_if.mean(), 1), "lstm_ae": round(100 * flags_ae.mean(), 1)})
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, sharey=True)
for ax, (name, fl) in zip(axes, [("Rules", flags_rules.values), ("Isolation Forest", flags_if), ("LSTM-AE", flags_ae)]):
    ax.plot(sm.index, sm, lw=0.4, color="tab:blue"); ax.plot(sm.index, sm.where(fl), ".", ms=1.5, color="tab:red"); ax.set_ylabel(name, fontsize=8)
axes[0].set_title("Detectors applied to the real soil-moisture deployment (red = flagged)")
axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3)); axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
fig.savefig(OUT_F / "fig7_real_deployment_detection.png"); fig.savefig(OUT_F / "fig7_real_deployment_detection.pdf"); plt.close(fig)

print(sorted(p.name for p in OUT_F.glob("*.png"))); print("done")
