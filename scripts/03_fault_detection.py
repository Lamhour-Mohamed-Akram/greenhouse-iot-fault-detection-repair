# %% [markdown]
# # Script 03 — Fault Detection Benchmark
#
# Benchmarks three unsupervised/semi-supervised detectors using the controlled
# 162-scenario injection corpus, with evaluation restricted to the 72 scenarios
# from held-out screened-clean windows, then applies the detectors to the real
# deployment.
#
# | Detector | Type | Training data |
# |----------|------|---------------|
# | **Rules** | OOR + stuck-run + Hampel + missingness | none (thresholds only) |
# | **Isolation Forest** | feature-based novelty detection | clean training windows |
# | **LSTM-autoencoder** | reconstruction-error novelty detection | clean training windows |
#
# Leakage control: per sensor, the first two clean windows are used to train the
# learned detectors; evaluation uses only scenarios injected into the *remaining*
# windows. All detectors are evaluated on the same test scenarios.
#
# Dropout (NaN) is trivially detectable from missingness, so all detectors share a
# missingness flag; the interesting comparison is on the value-corrupting faults.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

BASE = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
RESULTS = BASE / "results"
FIG, TAB, INJ = RESULTS / "figures", RESULTS / "tables", RESULTS / "injected"
MODELS = RESULTS / "models"
MODELS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

SEED = 42
np.random.seed(SEED)
VALID_RANGE = {"humidity": (0, 100), "temperature": (-5, 55), "soil_moisture": (0, 100)}
STUCK_THRESH_MIN = 60
HAMPEL_K, HAMPEL_WIN = 6, 61
SEQ_LEN = 60
N_TRAIN_WINDOWS = 2

data = pd.read_parquet(INJ / "injected_scenarios.parquet")
mf = pd.read_csv(INJ / "injection_manifest.csv", parse_dates=["window_start"])
win_df = pd.read_csv(TAB / "clean_windows.csv", parse_dates=["start", "end"])

train_windows, test_windows = {}, {}
for s, g in win_df.groupby("sensor"):
    starts = sorted(g.start)
    n_train = max(1, min(N_TRAIN_WINDOWS, len(starts) - 1))  # keep >=1 test window
    train_windows[s] = starts[:n_train]
    test_windows[s] = starts[n_train:]
test_ids = mf[mf.apply(lambda r: pd.Timestamp(r.window_start) in test_windows[r.sensor],
                       axis=1)].scenario_id.values
print(f"Test scenarios: {len(test_ids)} / {len(mf)} "
      f"(train windows per sensor: { {s: len(v) for s, v in train_windows.items()} })")

# %% [markdown]
# ## 1. Clean training data per sensor

# %%
clean_train = {}
for s in VALID_RANGE:
    parts = []
    for t0 in train_windows[s]:
        sc = mf[(mf.sensor == s) & (mf.window_start == t0)].scenario_id.iloc[0]
        d = data[data.scenario_id == sc]
        parts.append(pd.Series(d.value_clean.values, index=pd.DatetimeIndex(d.timestamp)))
    clean_train[s] = pd.concat(parts)
    print(s, "train samples:", len(clean_train[s]))

train_stats = {s: (x.mean(), x.std()) for s, x in clean_train.items()}

# diurnal profile from training windows (per minute-of-day, smoothed) —
# used as an Isolation Forest feature to expose bias/drift faults
profiles = {}
for s, x in clean_train.items():
    prof = x.groupby(x.index.hour * 60 + x.index.minute).mean()
    prof = prof.reindex(range(1440)).interpolate(limit_direction="both")
    profiles[s] = prof.rolling(31, center=True, min_periods=1).mean()

# %% [markdown]
# ## 2. Detector 1 — rule-based screening

# %%
def hampel_flags(x, k=HAMPEL_K, win=HAMPEL_WIN):
    med = x.rolling(win, center=True, min_periods=10).median()
    mad = (x - med).abs().rolling(win, center=True, min_periods=10).median()
    sigma = 1.4826 * mad
    return ((x - med).abs() > k * sigma) & sigma.gt(0) & x.notna()

def stuck_flags(x, thresh=STUCK_THRESH_MIN):
    grp = (x.diff() != 0).cumsum()
    runlen = x.groupby(grp).transform("size")
    return (runlen > thresh) & x.notna()

def detect_rules(x, sensor):
    lo, hi = VALID_RANGE[sensor]
    oor = ((x < lo) | (x > hi)) & x.notna()
    return (oor | stuck_flags(x) | hampel_flags(x) | x.isna()).values

# %% [markdown]
# ## 3. Detector 2 — Isolation Forest on engineered features

# %%
def features(x, sensor):
    mu, sd = train_stats[sensor]
    minute = x.index.hour * 60 + x.index.minute
    prof = profiles[sensor].reindex(minute).values
    grp = (x.diff() != 0).cumsum()
    runlen = x.groupby(grp).cumcount() + 1  # minutes the value has been constant
    f = pd.DataFrame({
        "z": (x - mu) / sd,
        "diff1": x.diff(),
        "roll_std": x.rolling(30, min_periods=5).std(),
        "profile_dev": x.values - prof,
        "runlen": runlen,
    }, index=x.index)
    return f.ffill().bfill()

iforests = {}
for s in VALID_RANGE:
    Xtr = features(clean_train[s], s)
    iforests[s] = IsolationForest(n_estimators=200, contamination=0.01,
                                  random_state=SEED).fit(Xtr)

def detect_iforest(x, sensor):
    pred = iforests[sensor].predict(features(x, sensor))
    return (pred == -1) | x.isna().values

# %% [markdown]
# ## 4. Detector 3 — LSTM-autoencoder

# %%
import tensorflow as tf

tf.random.set_seed(SEED)

def make_sequences(z, seq_len=SEQ_LEN):
    v = z.values.astype("float32")
    idx = np.arange(seq_len, len(v) + 1)
    return np.stack([v[i - seq_len:i] for i in idx])[..., None], idx - 1

def build_ae():
    inp = tf.keras.Input((SEQ_LEN, 1))
    h = tf.keras.layers.LSTM(32)(inp)
    h = tf.keras.layers.RepeatVector(SEQ_LEN)(h)
    h = tf.keras.layers.LSTM(32, return_sequences=True)(h)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(h)
    m = tf.keras.Model(inp, out)
    m.compile("adam", "mse")
    return m

def per_timestep_error(ae, z, n):
    """Squared reconstruction error per timestep, averaged over every sliding
    window that covers it (standard point-wise attribution for AE detectors)."""
    X, ends = make_sequences(z)
    se = (ae.predict(X, verbose=0) - X) ** 2  # (n_seq, SEQ_LEN, 1)
    err_sum, cnt = np.zeros(n), np.zeros(n)
    for k, end in enumerate(ends):
        err_sum[end - SEQ_LEN + 1:end + 1] += se[k, :, 0]
        cnt[end - SEQ_LEN + 1:end + 1] += 1
    return err_sum / np.maximum(cnt, 1)

autoencoders, ae_thresholds = {}, {}
for s in VALID_RANGE:
    mu, sd = train_stats[s]
    z = (clean_train[s] - mu) / sd
    X, _ = make_sequences(z)
    ae = build_ae()
    ae.fit(X, X, epochs=15, batch_size=256, verbose=0,
           validation_split=0.1, shuffle=True)
    err_t = per_timestep_error(ae, z, len(z))
    ae_thresholds[s] = float(np.quantile(err_t, 0.995))
    autoencoders[s] = ae
    ae.save(MODELS / f"lstm_ae_{s}.keras")
    print(f"{s}: AE trained on {len(X)} seqs, threshold={ae_thresholds[s]:.4f}")

def detect_ae(x, sensor):
    mu, sd = train_stats[sensor]
    z = ((x - mu) / sd).ffill().bfill()
    err_t = per_timestep_error(autoencoders[sensor], z, len(x))
    return (err_t > ae_thresholds[sensor]) | x.isna().values

# %% [markdown]
# ## 5. Benchmark on test scenarios

# %%
DETECTORS = {"rules": detect_rules, "iforest": detect_iforest, "lstm_ae": detect_ae}

rows = []
for sc in test_ids:
    d = data[data.scenario_id == sc]
    meta = mf[mf.scenario_id == sc].iloc[0]
    x = pd.Series(d.value_faulty.values, index=pd.DatetimeIndex(d.timestamp))
    truth = d.is_fault.values
    for name, fn in DETECTORS.items():
        pred = fn(x, meta.sensor)
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn_ = int((~pred & truth).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn_) if tp + fn_ else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"scenario_id": sc, "sensor": meta.sensor,
                     "fault_type": meta.fault_type, "severity": meta.severity,
                     "detector": name, "precision": prec, "recall": rec, "f1": f1,
                     "event_detected": bool((pred & truth).any())})
res = pd.DataFrame(rows)
res.to_csv(TAB / "detection_benchmark_raw.csv", index=False)

agg = (res.groupby(["detector", "fault_type"])[["precision", "recall", "f1"]]
       .mean().round(3))
agg.to_csv(TAB / "detection_benchmark_by_fault.csv")
print(agg)

# %%
sev = (res.groupby(["detector", "fault_type", "severity"]).f1.mean().round(3)
       .unstack().reindex(columns=["low", "medium", "high"]))
sev.to_csv(TAB / "detection_benchmark_by_severity.csv")
event = (res.groupby(["detector", "fault_type"]).event_detected.mean().round(3)
         .unstack())
event.to_csv(TAB / "detection_event_recall.csv")
print(event)

# %% [markdown]
# ## 6. Figures

# %%
# Fig 5 — F1 per fault type and detector
f1p = agg["f1"].unstack(0)[["rules", "iforest", "lstm_ae"]]
ax = f1p.plot(kind="bar", figsize=(7.5, 3.2),
              color=["tab:gray", "tab:blue", "tab:red"])
ax.set_ylabel("Point-wise F1 (mean over test scenarios)")
ax.set_xlabel("")
ax.set_ylim(0, 1.05)
ax.legend(["Rules", "Isolation Forest", "LSTM-AE"], fontsize=8)
ax.set_title("Detection performance per injected fault class")
plt.xticks(rotation=0)
plt.gcf().savefig(FIG / "fig5_detection_f1_by_fault.png")
plt.gcf().savefig(FIG / "fig5_detection_f1_by_fault.pdf")
plt.close()

# Fig 6 — severity effect (F1 heatmap)
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
for ax, det in zip(axes, ["rules", "iforest", "lstm_ae"]):
    m = sev.loc[det]
    im = ax.imshow(m.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(3), ["low", "med", "high"])
    ax.set_yticks(range(len(m)), m.index)
    ax.set_title(det, fontsize=9)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=axes, shrink=0.8, label="F1")
fig.suptitle("F1 by fault severity", y=1.02)
fig.savefig(FIG / "fig6_detection_severity_heatmap.png")
fig.savefig(FIG / "fig6_detection_severity_heatmap.pdf")
plt.close(fig)
print("Saved fig5, fig6")

# %% [markdown]
# ## 7. Application to the real deployment (soil moisture)
#
# The learned detectors, trained only on clean windows, are run over the entire
# real soil-moisture series — including the 8-day negative episode and the 153 h
# stuck episode that Script 01 characterized.

# %%
raw = pd.read_csv(BASE / "data" / "data.csv.gz", encoding="latin-1")
raw.columns = ["Id", "humidity", "temperature", "soil_moisture", "soil_temp",
               "co2", "light", "date"]
raw["date"] = pd.to_datetime(raw["date"])
sm = raw.sort_values("date").set_index("date")["soil_moisture"].resample("1min").median()

flags_rules = detect_rules(sm, "soil_moisture")
flags_ae = detect_ae(sm, "soil_moisture")
flags_if = detect_iforest(sm, "soil_moisture")
real_cov = pd.DataFrame({
    "detector": ["rules", "iforest", "lstm_ae"],
    "flagged_pct": [100 * f.mean() for f in (flags_rules, flags_if, flags_ae)]}).round(2)
real_cov.to_csv(TAB / "real_deployment_flagged_pct.csv", index=False)
print(real_cov)

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, sharey=True)
for ax, (name, fl) in zip(axes, [("Rules", flags_rules),
                                 ("Isolation Forest", flags_if),
                                 ("LSTM-AE", flags_ae)]):
    ax.plot(sm.index, sm, lw=0.4, color="tab:blue")
    ax.plot(sm.index, sm.where(fl), ".", ms=1.5, color="tab:red")
    ax.set_ylabel(name, fontsize=8)
axes[0].set_title("Detectors applied to the real soil-moisture deployment "
                  "(red = flagged)")
fig.savefig(FIG / "fig7_real_deployment_detection.png")
fig.savefig(FIG / "fig7_real_deployment_detection.pdf")
plt.close(fig)
print("Saved fig7")

# %% [markdown]
# ## 8. Headline numbers

# %%
headline = {
    "n_test_scenarios": int(len(test_ids)),
    "macro_f1": {d: round(float(res[res.detector == d].f1.mean()), 3)
                 for d in DETECTORS},
    "event_recall": {d: round(float(res[res.detector == d].event_detected.mean()), 3)
                     for d in DETECTORS},
    "best_detector_overall": res.groupby("detector").f1.mean().idxmax(),
    "real_soil_moisture_flagged_pct": {r.detector: r.flagged_pct
                                       for r in real_cov.itertuples()},
}
with open(TAB / "detection_headline.json", "w") as f:
    json.dump(headline, f, indent=2)
print(json.dumps(headline, indent=2))
