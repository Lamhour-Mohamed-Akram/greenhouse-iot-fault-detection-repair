# Script 15 - Window-length sensitivity of the three detectors
#
# One window parameter is varied at a time, all thresholds kept at their defaults:
#   - LSTM autoencoder sequence length 30 / 60 / 120 min (each retrained on the clean training windows,
#     seed 42, 15 epochs, threshold = 99.5th percentile of clean-training errors)
#   - Hampel window of the rule screen 31 / 61 / 121 min (k = 6)
#   - rolling-deviation window of the Isolation Forest features 15 / 30 / 60 min (contamination 1 %)
# Scored on the 72 held-out injection scenarios (macro-F1 over all six classes and over the five
# value-corrupting classes, per-class F1) and on the non-injected held-out clean windows
# (false-positive rate).
import json
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.ensemble import IsolationForest

tf.config.experimental.enable_op_determinism()
ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
DATA = ROOT / "data" / "data.csv.gz"
RES = ROOT / "results"; OUT_T = RES / "tables"; OUT_F = RES / "figures"
TAB, INJ, MODELS, OFIG, ORIG_T, MS_F = OUT_T, RES / "injected", RES / "models", OUT_F, OUT_T, OUT_F
OUT_T.mkdir(parents=True, exist_ok=True); OUT_F.mkdir(parents=True, exist_ok=True)
SEED = 42
VALID = {"humidity": (0, 100), "temperature": (-5, 55), "soil_moisture": (0, 100)}

data = pd.read_parquet(INJ / "injected_scenarios.parquet")
mf = pd.read_csv(INJ / "injection_manifest.csv", parse_dates=["window_start"])
win_df = pd.read_csv(TAB / "clean_windows.csv", parse_dates=["start", "end"])
train_w, test_w = {}, {}
for s, g in win_df.groupby("sensor"):
    st = sorted(g.start); n = max(1, min(2, len(st) - 1)); train_w[s] = st[:n]; test_w[s] = st[n:]
test_ids = mf[mf.apply(lambda r: pd.Timestamp(r.window_start) in test_w[r.sensor], axis=1)].scenario_id.values
def window_series(s, t0):
    sc = mf[(mf.sensor == s) & (mf.window_start == t0)].scenario_id.iloc[0]; d = data[data.scenario_id == sc]
    return pd.Series(d.value_clean.values, index=pd.DatetimeIndex(d.timestamp))
clean_train = {s: pd.concat([window_series(s, t) for t in train_w[s]]) for s in VALID}
clean_test = {s: [window_series(s, t) for t in test_w[s]] for s in VALID}
stats = {s: (x.mean(), x.std()) for s, x in clean_train.items()}
profiles = {}
for s, x in clean_train.items():
    pr = x.groupby(x.index.hour * 60 + x.index.minute).mean().reindex(range(1440)).interpolate(limit_direction="both")
    profiles[s] = pr.rolling(31, center=True, min_periods=1).mean()

def hampel(x, k=6, win=61):
    med = x.rolling(win, center=True, min_periods=10).median(); mad = (x - med).abs().rolling(win, center=True, min_periods=10).median()
    return ((x - med).abs() > k * 1.4826 * mad) & mad.gt(0) & x.notna()
def stuck(x, thr=60):
    grp = (x.diff() != 0).cumsum(); return (x.groupby(grp).transform("size") > thr) & x.notna()
def rules(x, s, win=61):
    lo, hi = VALID[s]; return ((((x < lo) | (x > hi)) & x.notna()) | stuck(x) | hampel(x, 6, win) | x.isna()).values
def feats(x, s, roll=30):
    mu, sd = stats[s]; pr = profiles[s].reindex(x.index.hour * 60 + x.index.minute).values
    grp = (x.diff() != 0).cumsum()
    return pd.DataFrame({"z": (x - mu) / sd, "diff1": x.diff(), "roll_std": x.rolling(roll, min_periods=5).std(),
                         "profile_dev": x.values - pr, "runlen": x.groupby(grp).cumcount() + 1}, index=x.index).ffill().bfill()
def seqs(z, L):
    v = z.values.astype("float32"); idx = np.arange(L, len(v) + 1)
    return np.stack([v[i - L:i] for i in idx])[..., None], idx - 1
def pte(ae, z, n, L):
    X, ends = seqs(z, L); se = (ae.predict(X, verbose=0) - X) ** 2; es, c = np.zeros(n), np.zeros(n)
    for k, e in enumerate(ends): es[e - L + 1:e + 1] += se[k, :, 0]; c[e - L + 1:e + 1] += 1
    return es / np.maximum(c, 1)
def build_ae(L):
    inp = tf.keras.Input((L, 1)); h = tf.keras.layers.LSTM(32)(inp); h = tf.keras.layers.RepeatVector(L)(h)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(tf.keras.layers.LSTM(32, return_sequences=True)(h))
    m = tf.keras.Model(inp, out); m.compile("adam", "mse"); return m
def prf(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

detectors = {}
for win in [31, 61, 121]: detectors[("rules", f"Hampel window {win} min")] = lambda x, s, win=win: rules(x, s, win)
for roll in [15, 30, 60]:
    ifs = {s: IsolationForest(n_estimators=200, contamination=0.01, random_state=SEED).fit(feats(clean_train[s], s, roll)) for s in VALID}
    detectors[("iforest", f"rolling window {roll} min")] = lambda x, s, ifs=ifs, roll=roll: (ifs[s].predict(feats(x, s, roll)) == -1) | x.isna().values
for L in [30, 60, 120]:
    aes, thr = {}, {}
    for s in VALID:
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
        mu, sd = stats[s]; z = (clean_train[s] - mu) / sd; X, _ = seqs(z, L)
        ae = build_ae(L); ae.fit(X, X, epochs=15, batch_size=256, verbose=0, validation_split=0.1, shuffle=True)
        aes[s] = ae; thr[s] = float(np.quantile(pte(ae, z, len(z), L), 0.995))
    def det(x, s, aes=aes, thr=thr, L=L):
        mu, sd = stats[s]; return (pte(aes[s], ((x - mu) / sd).ffill().bfill(), len(x), L) > thr[s]) | x.isna().values
    detectors[("lstm_ae", f"sequence {L} min")] = det
    print("AE trained, L =", L)

rows, fps = [], []
for (dname, cfg), fn in detectors.items():
    for sc in test_ids:
        d = data[data.scenario_id == sc]; meta = mf[mf.scenario_id == sc].iloc[0]
        x = pd.Series(d.value_faulty.values, index=pd.DatetimeIndex(d.timestamp))
        rows.append({"detector": dname, "setting": cfg, "fault_type": meta.fault_type, "f1": prf(fn(x, meta.sensor), d.is_fault.values)})
    for s in VALID:
        for x in clean_test[s]: fps.append({"detector": dname, "setting": cfg, "fp": float(fn(x, s).mean())})
    print(dname, cfg, "done")
r = pd.DataFrame(rows); f = pd.DataFrame(fps)
out = r.groupby(["detector", "setting"]).apply(lambda g: pd.Series({
    "macro_f1": g.f1.mean(), "macro_f1_5": g[g.fault_type != "dropout"].f1.mean(),
    **{f"f1_{t}": g[g.fault_type == t].f1.mean() for t in ["bias", "drift", "noise", "spike", "stuck"]}}))
out = out.join(f.groupby(["detector", "setting"]).fp.agg(fp_mean="mean", fp_max="max")).round(4)
out.to_csv(OUT_T / "detection_window_length_sensitivity.csv"); print(out.to_string()); print("done")
