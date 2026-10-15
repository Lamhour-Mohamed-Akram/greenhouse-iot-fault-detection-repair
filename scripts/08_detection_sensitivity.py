# Script 08 - Detection benchmark: uncertainty, sensitivity and near-threshold scenarios
#
# Reuses the injection corpus of script 02 and the autoencoders saved by script 03.
#   - per-class F1 dispersion across scenarios and across held-out windows
#   - macro-F1 over the five value-corrupting classes
#   - false-positive rates on non-injected held-out clean windows
#   - threshold sensitivity: Isolation Forest contamination, autoencoder quantile, Hampel k
#   - near-threshold stuck episodes (30/60/90 min) and quasi-stuck (jitter) scenarios
#   - injection amplitudes in native units and share of samples crossing physical bounds
#   - rule coverage of the real soil-moisture stream with the raw-sample stuck definition
#   - number of clean 24-h windows under different Hampel k
import json
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
DATA = ROOT / "data" / "data.csv.gz"
RES = ROOT / "results"; OUT_T = RES / "tables"; OUT_F = RES / "figures"
TAB, INJ, MODELS, OFIG, ORIG_T, MS_F = OUT_T, RES / "injected", RES / "models", OUT_F, OUT_T, OUT_F
OUT_T.mkdir(parents=True, exist_ok=True); OUT_F.mkdir(parents=True, exist_ok=True)
SEED = 42; np.random.seed(SEED); tf.random.set_seed(SEED)
VALID = {"humidity": (0, 100), "temperature": (-5, 55), "soil_moisture": (0, 100)}
QUANT = {"humidity": 0.1, "temperature": 0.1, "soil_moisture": 1.0}     # resolution of the 1-min medians
STUCK_MIN = 60; HAMPEL_WIN = 61; SEQ_LEN = 60; N_TRAIN = 2

data = pd.read_parquet(INJ / "injected_scenarios.parquet")
mf = pd.read_csv(INJ / "injection_manifest.csv", parse_dates=["window_start"])
win_df = pd.read_csv(TAB / "clean_windows.csv", parse_dates=["start", "end"])
train_w, test_w = {}, {}
for s, g in win_df.groupby("sensor"):
    st = sorted(g.start); n = max(1, min(N_TRAIN, len(st) - 1)); train_w[s] = st[:n]; test_w[s] = st[n:]
test_ids = mf[mf.apply(lambda r: pd.Timestamp(r.window_start) in test_w[r.sensor], axis=1)].scenario_id.values
assert len(test_ids) == 72

clean_train, clean_test = {}, {}
for s in VALID:
    parts = []
    for t0 in train_w[s]:
        sc = mf[(mf.sensor == s) & (mf.window_start == t0)].scenario_id.iloc[0]; d = data[data.scenario_id == sc]
        parts.append(pd.Series(d.value_clean.values, index=pd.DatetimeIndex(d.timestamp)))
    clean_train[s] = pd.concat(parts)
    clean_test[s] = []
    for t0 in test_w[s]:
        sc = mf[(mf.sensor == s) & (mf.window_start == t0)].scenario_id.iloc[0]; d = data[data.scenario_id == sc]
        clean_test[s].append(pd.Series(d.value_clean.values, index=pd.DatetimeIndex(d.timestamp)))
train_stats = {s: (x.mean(), x.std()) for s, x in clean_train.items()}
profiles = {}
for s, x in clean_train.items():
    prof = x.groupby(x.index.hour * 60 + x.index.minute).mean().reindex(range(1440)).interpolate(limit_direction="both")
    profiles[s] = prof.rolling(31, center=True, min_periods=1).mean()

# ---------------------------------------------------------------- detectors (as in script 03, parameterized)
def hampel_flags(x, k=6, win=HAMPEL_WIN):
    med = x.rolling(win, center=True, min_periods=10).median(); mad = (x - med).abs().rolling(win, center=True, min_periods=10).median()
    return ((x - med).abs() > k * 1.4826 * mad) & mad.gt(0) & x.notna()
def stuck_flags(x, thresh=STUCK_MIN):
    grp = (x.diff() != 0).cumsum(); runlen = x.groupby(grp).transform("size"); return (runlen > thresh) & x.notna()
def detect_rules(x, s, k=6):
    lo, hi = VALID[s]; oor = ((x < lo) | (x > hi)) & x.notna()
    return (oor | stuck_flags(x) | hampel_flags(x, k) | x.isna()).values
def features(x, s):
    mu, sd = train_stats[s]; minute = x.index.hour * 60 + x.index.minute; prof = profiles[s].reindex(minute).values
    grp = (x.diff() != 0).cumsum(); runlen = x.groupby(grp).cumcount() + 1
    f = pd.DataFrame({"z": (x - mu) / sd, "diff1": x.diff(), "roll_std": x.rolling(30, min_periods=5).std(),
                      "profile_dev": x.values - prof, "runlen": runlen}, index=x.index)
    return f.ffill().bfill()
IF_CONTAM = [0.005, 0.01, 0.02, 0.05]
iforests = {c: {s: IsolationForest(n_estimators=200, contamination=c, random_state=SEED).fit(features(clean_train[s], s)) for s in VALID} for c in IF_CONTAM}
def detect_if(x, s, c=0.01):
    return (iforests[c][s].predict(features(x, s)) == -1) | x.isna().values

def make_sequences(z, seq_len=SEQ_LEN):
    v = z.values.astype("float32"); idx = np.arange(seq_len, len(v) + 1)
    return np.stack([v[i - seq_len:i] for i in idx])[..., None], idx - 1
def per_timestep_error(ae, z, n):
    X, ends = make_sequences(z); se = (ae.predict(X, verbose=0) - X) ** 2
    es, c = np.zeros(n), np.zeros(n)
    for k, end in enumerate(ends): es[end - SEQ_LEN + 1:end + 1] += se[k, :, 0]; c[end - SEQ_LEN + 1:end + 1] += 1
    return es / np.maximum(c, 1)
AE_Q = [0.99, 0.995, 0.999]
autoencoders, ae_thr = {}, {}
for s in VALID:
    ae = tf.keras.models.load_model(MODELS / f"lstm_ae_{s}.keras", compile=False); autoencoders[s] = ae
    mu, sd = train_stats[s]; z = (clean_train[s] - mu) / sd; err = per_timestep_error(ae, z, len(z))
    ae_thr[s] = {q: float(np.quantile(err, q)) for q in AE_Q}
    print(s, "AE thresholds", ae_thr[s])
ae_err_cache = {}
def ae_errors(x, s):
    key = (id(x), s)
    mu, sd = train_stats[s]; z = ((x - mu) / sd).ffill().bfill(); return per_timestep_error(autoencoders[s], z, len(x))
def detect_ae(x, s, q=0.995, err=None):
    err = ae_errors(x, s) if err is None else err; return (err > ae_thr[s][q]) | x.isna().values

def prf(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

# ---------------------------------------------------------------- benchmark on the 72 held-out scenarios, all configurations
CONFIGS = ([("rules", f"k={k}", lambda x, s, k=k: detect_rules(x, s, k)) for k in [4, 6, 8]] +
           [("iforest", f"c={c}", lambda x, s, c=c: detect_if(x, s, c)) for c in IF_CONTAM])
rows = []
for sc in test_ids:
    d = data[data.scenario_id == sc]; meta = mf[mf.scenario_id == sc].iloc[0]
    x = pd.Series(d.value_faulty.values, index=pd.DatetimeIndex(d.timestamp)); truth = d.is_fault.values
    err = ae_errors(x, meta.sensor)
    cfgs = CONFIGS + [("lstm_ae", f"q={q}", lambda x, s, q=q, err=err: detect_ae(x, s, q, err)) for q in AE_Q]
    for det, cfg, fn in cfgs:
        pred = fn(x, meta.sensor); p, r, f1 = prf(pred, truth)
        rows.append({"scenario_id": sc, "sensor": meta.sensor, "window_start": meta.window_start, "fault_type": meta.fault_type,
                     "severity": meta.severity, "detector": det, "config": cfg, "precision": p, "recall": r, "f1": f1,
                     "event_detected": bool((pred & truth).any())})
res = pd.DataFrame(rows); res.to_csv(OUT_T / "detection_sensitivity_raw.csv", index=False)
DEFAULT = {"rules": "k=6", "iforest": "c=0.01", "lstm_ae": "q=0.995"}
base = res[res.apply(lambda r: r.config == DEFAULT[r.detector], axis=1)]
# consistency check against script 03
macro = base.groupby("detector").f1.mean().round(3).to_dict(); print("macro-F1 (reproduction, should be rules .444 / iforest .333 / lstm_ae .356):", macro)
macro_nodrop = base[base.fault_type != "dropout"].groupby("detector").f1.mean().round(3).to_dict()
# window-clustered dispersion
byw = base.groupby(["detector", "fault_type", "window_start"]).f1.mean().reset_index()
clus = byw.groupby(["detector", "fault_type"]).f1.agg(["mean", "std", "min", "max", "count"]).round(3)
clus.to_csv(OUT_T / "detection_f1_by_window_cluster.csv")
sc_sd = base.groupby(["detector", "fault_type"]).f1.agg(["mean", "std"]).round(3)
comb = sc_sd.join(clus, lsuffix="_scenario", rsuffix="_window"); comb.to_csv(OUT_T / "detection_f1_dispersion_comparison.csv")
print(comb)
# paired detector differences per scenario : bootstrap over windows
pair_rows = []
rng = np.random.default_rng(0)
for ft in base.fault_type.unique():
    piv = base[base.fault_type == ft].pivot_table(index=["window_start", "severity"], columns="detector", values="f1")
    for a, b in [("rules", "lstm_ae"), ("rules", "iforest"), ("lstm_ae", "iforest")]:
        diff = (piv[a] - piv[b]); wins = diff.groupby(level=0).mean()  # per-window mean difference
        boots = [wins.sample(len(wins), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
        pair_rows.append({"fault_type": ft, "a": a, "b": b, "mean_diff": round(float(diff.mean()), 3), "n_windows": int(len(wins)),
                          "ci_lo": round(float(np.percentile(boots, 2.5)), 3), "ci_hi": round(float(np.percentile(boots, 97.5)), 3)})
pd.DataFrame(pair_rows).to_csv(OUT_T / "detection_paired_differences.csv", index=False)
# threshold sensitivity summary
sens = res.groupby(["detector", "config"]).apply(lambda g: pd.Series({
    "macro_f1": g.f1.mean(), "macro_f1_no_dropout": g[g.fault_type != "dropout"].f1.mean(),
    **{f"f1_{ft}": g[g.fault_type == ft].f1.mean() for ft in ["bias", "drift", "noise", "spike", "stuck"]}})).round(3)
# false-positive rate on non-injected held-out clean windows
fp_rows = []
for s in VALID:
    for x in clean_test[s]:
        err = ae_errors(x, s)
        for det, cfg, fn in CONFIGS + [("lstm_ae", f"q={q}", lambda x, s, q=q, err=err: detect_ae(x, s, q, err)) for q in AE_Q]:
            fp_rows.append({"sensor": s, "window_start": x.index[0], "detector": det, "config": cfg, "fp_rate": float(fn(x, s).mean())})
fp = pd.DataFrame(fp_rows); fp.to_csv(OUT_T / "detection_fp_clean_windows_raw.csv", index=False)
fp_summary = fp.groupby(["detector", "config"]).fp_rate.agg(["mean", "max"]).round(4)
sens = sens.join(fp_summary.rename(columns={"mean": "fp_rate_clean_mean", "max": "fp_rate_clean_max"}))
sens.to_csv(OUT_T / "detection_threshold_sensitivity.csv"); print(sens)

# ---------------------------------------------------------------- near-threshold stuck and quasi-stuck scenarios
rng = np.random.default_rng(SEED)
nt_rows = []
for s in VALID:
    for x in clean_test[s]:
        err_clean = None
        for kind, dur in [("stuck", 30), ("stuck", 60), ("stuck", 90), ("quasi_stuck", 240), ("quasi_stuck", 480)]:
            for rep in range(3):
                start = int(rng.integers(0, len(x) - dur)); y = x.copy(); m = np.zeros(len(x), bool)
                if kind == "stuck":
                    y.iloc[start:start + dur] = y.iloc[start]
                else:  # constant level plus jitter of one quantization step (value never exactly constant for > 60 min)
                    jit = rng.choice([-1, 0, 1], dur) * QUANT[s]
                    y.iloc[start:start + dur] = y.iloc[start] + jit
                m[start:start + dur] = True
                err = ae_errors(y, s)
                for det, fn in [("rules", lambda y, s: detect_rules(y, s, 6)), ("iforest", lambda y, s: detect_if(y, s, 0.01)),
                                ("lstm_ae", lambda y, s, err=err: detect_ae(y, s, 0.995, err))]:
                    pred = fn(y, s); p, r, f1 = prf(pred, m)
                    nt_rows.append({"sensor": s, "window_start": x.index[0], "kind": kind, "duration_min": dur, "rep": rep,
                                    "detector": det, "precision": p, "recall": r, "f1": f1, "event_detected": bool((pred & m).any())})
nt = pd.DataFrame(nt_rows); nt.to_csv(OUT_T / "detection_near_threshold_stuck_raw.csv", index=False)
nt_summary = nt.groupby(["kind", "duration_min", "detector"])[["precision", "recall", "f1", "event_detected"]].mean().round(3)
nt_summary.to_csv(OUT_T / "detection_near_threshold_stuck.csv"); print(nt_summary)

# ---------------------------------------------------------------- injection amplitudes in native units
amp_rows = []
K = {"spike": {"low": 3, "medium": 5, "high": 8}, "bias": {"low": 1, "medium": 2, "high": 3}, "drift": {"low": 1, "medium": 3, "high": 5},
     "noise": {"low": 0.5, "medium": 1.0, "high": 2.0}}
for s in VALID:
    sds = data[(data.sensor == s) & (data.fault_type == "spike") & (data.severity == "low")].groupby("scenario_id").value_clean.std()
    for ft, kk in K.items():
        for sev, k in kk.items():
            sub = data[(data.sensor == s) & (data.fault_type == ft) & (data.severity == sev) & data.is_fault]
            lo, hi = VALID[s]; cross = ((sub.value_faulty < lo) | (sub.value_faulty > hi)).mean()
            amp_rows.append({"sensor": s, "fault_type": ft, "severity": sev, "k_sigma": k, "window_sd_min": round(float(sds.min()), 3),
                             "window_sd_max": round(float(sds.max()), 3), "amplitude_native_min": round(float(k * sds.min()), 3),
                             "amplitude_native_max": round(float(k * sds.max()), 3), "pct_crossing_bounds": round(100 * float(cross), 1)})
amp = pd.DataFrame(amp_rows); amp.to_csv(OUT_T / "injection_amplitudes_native.csv", index=False); print(amp)
# temporal-pattern parameters of the injection protocol (from script 02)
inj_params = {"spike": "n = max(1, frac*1440) isolated samples, frac = 0.2/0.5/1.0 %, random sign, uniformly random positions",
              "stuck": "one episode of 90/240/480 min, random start, value frozen at the start sample",
              "drift": "linear ramp from 0 to k*sigma starting at a random position in the second quarter of the window and lasting to its end",
              "bias": "step of +/- k*sigma (random sign) from a random position in the second quarter of the window to its end",
              "noise": "Gaussian noise N(0, (k*sigma)^2) added over one 360-min episode starting in the first half of the window",
              "dropout": "one episode of 15/60/240 min set to missing, random start",
              "bounds": "injected values are not clipped; samples crossing the physical bounds remain labelled with their injected class"}

# ---------------------------------------------------------------- real-stream coverage with the raw-sample stuck definition
raw = pd.read_csv(DATA, encoding="latin-1"); raw.columns = ["Id", "humidity", "temperature", "soil_moisture", "soil_temp", "co2", "light", "date"]
raw["date"] = pd.to_datetime(raw["date"]); raw = raw.sort_values("date").reset_index(drop=True)
sm = raw.set_index("date")["soil_moisture"].resample("1min").median()
x = raw["soil_moisture"]; d = raw["date"]; gap = d.diff().dt.total_seconds() > 300
g = ((x.diff() != 0) | gap).cumsum(); st = d.groupby(g).transform("first"); en = d.groupby(g).transform("last")
raw_stuck = pd.Series(((en - st).dt.total_seconds() > 3600).values, index=d).groupby(pd.Grouper(freq="1min")).any().reindex(sm.index, fill_value=False)
lo, hi = VALID["soil_moisture"]; oor = ((sm < lo) | (sm > hi)) & sm.notna(); spk = hampel_flags(sm, 6); miss = sm.isna()
grid_stuck = stuck_flags(sm)
cov = {"rules_grid_stuck_pct": round(100 * float((oor | grid_stuck | spk | miss).mean()), 1),
       "rules_raw_stuck_pct": round(100 * float((oor | (raw_stuck & sm.notna()) | spk | miss).mean()), 1),
       "components_pct": {"missing_minutes": round(100 * float(miss.mean()), 1), "oor": round(100 * float(oor.mean()), 1),
                          "stuck_grid_not_oor": round(100 * float((grid_stuck & ~oor & ~miss).mean()), 1),
                          "stuck_raw_not_oor": round(100 * float((raw_stuck & sm.notna() & ~oor).mean()), 1), "spike": round(100 * float(spk.mean()), 2)},
       "table4_total_soil_moisture_pct": 51.6}
print(cov)

# ---------------------------------------------------------------- clean-screen sensitivity to Hampel k
def clean_mask(x, lo, hi, k):
    ok = x.notna() & (x >= lo) & (x <= hi)
    grp = (x.diff() != 0).cumsum(); runlen = x.groupby(grp).transform("size"); ok &= ~((runlen > STUCK_MIN) & x.notna())
    med = x.rolling(61, center=True, min_periods=10).median(); mad = (x - med).abs().rolling(61, center=True, min_periods=10).median()
    spike = ((x - med).abs() > k * 1.4826 * mad) & mad.gt(0)
    ok &= ~spike.rolling(61, center=True, min_periods=1).max().astype(bool); return ok
g1 = raw.set_index("date")[list(VALID)].resample("1min").median()
cs_rows = []
for k in [4, 6, 8]:
    for s in VALID:
        ok = clean_mask(g1[s], *VALID[s], k); grp = (ok != ok.shift()).cumsum(); n24 = 0
        for _, seg in ok[ok].groupby(grp[ok]): n24 += len(seg) // 1440
        cs_rows.append({"hampel_k": k, "sensor": s, "clean_minutes_pct": round(100 * float(ok.mean()), 1), "n_24h_windows": int(n24)})
cs = pd.DataFrame(cs_rows); cs.to_csv(OUT_T / "clean_screen_sensitivity.csv", index=False); print(cs)

out = {"macro_f1": macro, "macro_f1_no_dropout": macro_nodrop, "n_held_out_windows": {s: len(test_w[s]) for s in VALID},
       "real_stream_coverage": cov, "injection_params": inj_params, "ae_thresholds": ae_thr}
with open(OUT_T / "detection_extended_headline.json", "w") as f: json.dump(out, f, indent=2, default=str)
print("done")
