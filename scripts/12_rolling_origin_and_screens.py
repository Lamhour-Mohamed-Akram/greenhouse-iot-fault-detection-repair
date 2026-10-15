# Script 12 - Clean-screen checks, rolling-origin evaluation and operational test set
#
#   (a) clean-screen sensitivity to the spike exclusion margin (15/30/60 min) and an
#       autoencoder-based screen (no rule criterion) on which the rule flag rate is measured
#   (b) rolling-origin downstream evaluation: three chronological test blocks, every fitted
#       component re-estimated on the data preceding the block, differences summarized across blocks
#   (c) operational test set: every observed target of the block, no rule screen, inputs as recorded
#       with missing slots forward-filled inside the block
import json
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from scipy import stats

tf.config.experimental.enable_op_determinism()
ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
DATA = ROOT / "data" / "data.csv.gz"
RES = ROOT / "results"; OUT_T = RES / "tables"; OUT_F = RES / "figures"
TAB, INJ, MODELS, OFIG, ORIG_T, MS_F = OUT_T, RES / "injected", RES / "models", OUT_F, OUT_T, OUT_F
OUT_T.mkdir(parents=True, exist_ok=True); OUT_F.mkdir(parents=True, exist_ok=True)
FEATURES = ["humidity", "temperature", "soil_moisture"]; TARGET = "soil_moisture"
VALID = {"humidity": (0, 100), "temperature": (-5, 55), "soil_moisture": (0, 100)}
SEEDS = list(range(42, 52)); SEQ = 12; AE_SEQ = 12; LINEAR_MAX = 48; EPOCHS, BATCH = 30, 256

raw = pd.read_csv(DATA, encoding="latin-1")
raw.columns = ["Id", "humidity", "temperature", "soil_moisture", "soil_temp", "co2", "light", "date"]
raw["date"] = pd.to_datetime(raw["date"]); raw = raw.sort_values("date").reset_index(drop=True)

# =====================================================================  (a) clean-screen sensitivity (1-min grid)
g1 = raw.set_index("date")[FEATURES].resample("1min").median()
def hampel(x, k=6, win=61):
    med = x.rolling(win, center=True, min_periods=10).median(); mad = (x - med).abs().rolling(win, center=True, min_periods=10).median()
    return ((x - med).abs() > k * 1.4826 * mad) & mad.gt(0) & x.notna()
def stuck_grid(x, thresh=60):
    grp = (x.diff() != 0).cumsum(); runlen = x.groupby(grp).transform("size"); return (runlen > thresh) & x.notna()
def clean_mask(x, lo, hi, k=6, margin=30):
    ok = x.notna() & (x >= lo) & (x <= hi) & ~stuck_grid(x)
    spike = hampel(x, k)
    ok &= ~spike.rolling(2 * margin + 1, center=True, min_periods=1).max().astype(bool); return ok
def windows_of(ok, max_per=4):
    grp = (ok != ok.shift()).cumsum(); out = []
    for _, seg in ok[ok].groupby(grp[ok]):
        t = seg.index[0]
        while t + pd.Timedelta(minutes=1439) <= seg.index[-1]:
            out.append(t); t += pd.Timedelta(minutes=1440)
    return out[:max_per]
def raw_stuck_1min(s):
    xx = raw[s]; d = raw["date"]; gap = d.diff().dt.total_seconds() > 300
    gg = ((xx.diff() != 0) | gap).cumsum(); st = d.groupby(gg).transform("first"); en = d.groupby(gg).transform("last")
    return pd.Series(((en - st).dt.total_seconds() > 3600).values, index=d).groupby(pd.Grouper(freq="1min")).any().reindex(g1.index, fill_value=False)
RAW_STUCK_1M = {s: raw_stuck_1min(s) for s in FEATURES}
def rules_flags(x, s, k=6, stuck="grid"):
    lo, hi = VALID[s]; stk = stuck_grid(x) if stuck == "grid" else (RAW_STUCK_1M[s].reindex(x.index) & x.notna())
    return (((x < lo) | (x > hi)) & x.notna()) | stk | hampel(x, k) | x.isna()

# learned detectors exactly as in script 03 (saved AE, IF refit on the same clean training windows)
from sklearn.ensemble import IsolationForest
data = pd.read_parquet(INJ / "injected_scenarios.parquet"); mf = pd.read_csv(INJ / "injection_manifest.csv", parse_dates=["window_start"])
win_df = pd.read_csv(TAB / "clean_windows.csv", parse_dates=["start", "end"])
train_w, test_w = {}, {}
for s, g in win_df.groupby("sensor"):
    st = sorted(g.start); n = max(1, min(2, len(st) - 1)); train_w[s] = st[:n]; test_w[s] = st[n:]
clean_train, train_stats, profiles, autoencoders, ae_thr, iforests = {}, {}, {}, {}, {}, {}
def pte(ae, z, n, seq=60):
    v = z.values.astype("float32"); idx = np.arange(seq, len(v) + 1); X = np.stack([v[i - seq:i] for i in idx])[..., None]
    se = (ae.predict(X, verbose=0) - X) ** 2; es, c = np.zeros(n), np.zeros(n)
    for k, end in enumerate(idx - 1): es[end - seq + 1:end + 1] += se[k, :, 0]; c[end - seq + 1:end + 1] += 1
    return es / np.maximum(c, 1)
for s in FEATURES:
    parts = []
    for t0 in train_w[s]:
        sc = mf[(mf.sensor == s) & (mf.window_start == t0)].scenario_id.iloc[0]; d = data[data.scenario_id == sc]
        parts.append(pd.Series(d.value_clean.values, index=pd.DatetimeIndex(d.timestamp)))
    ct = pd.concat(parts); clean_train[s] = ct; train_stats[s] = (ct.mean(), ct.std())
    prof = ct.groupby(ct.index.hour * 60 + ct.index.minute).mean().reindex(range(1440)).interpolate(limit_direction="both")
    profiles[s] = prof.rolling(31, center=True, min_periods=1).mean()
    ae = tf.keras.models.load_model(MODELS / f"lstm_ae_{s}.keras", compile=False); autoencoders[s] = ae
    ae_thr[s] = float(np.quantile(pte(ae, (ct - train_stats[s][0]) / train_stats[s][1], len(ct)), 0.995))
def features(x, s):
    mu, sd = train_stats[s]; minute = x.index.hour * 60 + x.index.minute; pr = profiles[s].reindex(minute).values
    grp = (x.diff() != 0).cumsum(); runlen = x.groupby(grp).cumcount() + 1
    return pd.DataFrame({"z": (x - mu) / sd, "diff1": x.diff(), "roll_std": x.rolling(30, min_periods=5).std(), "profile_dev": x.values - pr, "runlen": runlen}, index=x.index).ffill().bfill()
for s in FEATURES: iforests[s] = IsolationForest(n_estimators=200, contamination=0.01, random_state=42).fit(features(clean_train[s], s))
def ae_flags(x, s):
    mu, sd = train_stats[s]; return (pte(autoencoders[s], ((x - mu) / sd).ffill().bfill(), len(x)) > ae_thr[s]) | x.isna().values
def if_flags(x, s): return (iforests[s].predict(features(x, s)) == -1) | x.isna().values

rows = []
for margin in [15, 30, 60]:
    for s in FEATURES:
        ok = clean_mask(g1[s], *VALID[s], margin=margin); ws = windows_of(ok)
        held = [t for t in ws if t not in train_w[s]]   # windows other than the detector-training windows
        fps = {"rules": [], "iforest": [], "lstm_ae": []}
        for t0 in held:
            x = g1.loc[t0:t0 + pd.Timedelta(minutes=1439), s]
            fps["rules"].append(float(rules_flags(x, s).mean())); fps["iforest"].append(float(if_flags(x, s).mean())); fps["lstm_ae"].append(float(ae_flags(x, s).mean()))
        rows.append({"screen": f"rule screen, margin {margin} min", "sensor": s, "clean_minutes_pct": round(100 * float(ok.mean()), 1), "n_windows": len(ws), "n_eval_windows": len(held),
                     **{f"fp_{k}": (round(100 * float(np.mean(v)), 3) if v else np.nan) for k, v in fps.items()}})
# detector-independent screen: 24-h windows (present, in range) in which the LSTM-AE flags < 1 % of slots; no stuck or Hampel criterion
for s in FEATURES:
    x = g1[s]; present = x.notna() & (x >= VALID[s][0]) & (x <= VALID[s][1])
    cand = windows_of(present, max_per=100)
    aef = pd.Series(ae_flags(x, s), index=x.index)
    held = [t for t in cand if aef.loc[t:t + pd.Timedelta(minutes=1439)].mean() < 0.01 and t not in train_w[s]]
    fps = {"rules": [], "iforest": [], "lstm_ae": []}
    for t0 in held:
        xw = x.loc[t0:t0 + pd.Timedelta(minutes=1439)]
        fps["rules"].append(float(rules_flags(xw, s, stuck="raw").mean())); fps["iforest"].append(float(if_flags(xw, s).mean())); fps["lstm_ae"].append(float(ae_flags(xw, s).mean()))
        fps.setdefault("rules_gridstuck", []).append(float(rules_flags(xw, s, stuck="grid").mean()))
    rows.append({"screen": "LSTM-AE screen (< 1 % flagged, no rule criterion)", "sensor": s, "clean_minutes_pct": np.nan, "n_windows": len(held), "n_eval_windows": len(held),
                 **{f"fp_{k}": (round(100 * float(np.mean(v)), 3) if v else np.nan) for k, v in fps.items()},
                 "windows": ";".join(str(t) for t in held)})
cs = pd.DataFrame(rows); cs.to_csv(OUT_T / "clean_screen_margin_and_independent.csv", index=False); print(cs.drop(columns=["windows"], errors="ignore").to_string())
import os
if os.environ.get("SCREEN_ONLY"): raise SystemExit(0)

# =====================================================================  (b)+(c) downstream: rolling origin and operational test set (5-min grid, H = 1)
grid = raw.set_index("date")[FEATURES].resample("5min").median()
def raw_stuck_to_grid(s):
    x = raw[s]; d = raw["date"]; gap = d.diff().dt.total_seconds() > 300
    g = ((x.diff() != 0) | gap).cumsum(); st = d.groupby(g).transform("first"); en = d.groupby(g).transform("last")
    return pd.Series(((en - st).dt.total_seconds() > 3600).values, index=d).groupby(pd.Grouper(freq="5min")).any().reindex(grid.index, fill_value=False)
raw_stuck = {s: raw_stuck_to_grid(s) for s in FEATURES}
_STUCK_CACHE = {}
def raw_stuck_window(s, t0, t1, freq):
    """Raw-sample stuck runs (exact constancy > 1 h, split at > 5-min gaps) computed only from the raw
    samples in [t0, t1), so that no run crosses a partition boundary; mapped to the grid."""
    key = (s, t0, t1, freq)
    if key not in _STUCK_CACHE:
        r = raw[(raw["date"] >= t0) & (raw["date"] < t1)]
        x = r[s]; d = r["date"]; gap = d.diff().dt.total_seconds() > 300
        g = ((x.diff() != 0) | gap).cumsum(); st = d.groupby(g).transform("first"); en = d.groupby(g).transform("last")
        _STUCK_CACHE[key] = pd.Series(((en - st).dt.total_seconds() > 3600).values, index=d).groupby(pd.Grouper(freq=freq)).any()
    return _STUCK_CACHE[key]

def rule_mask5(x, s):
    lo, hi = VALID[s]; oor = ((x < lo) | (x > hi)) & x.notna(); stuck = raw_stuck_window(s, x.index[0], x.index[-1] + pd.Timedelta("5min"), "5min").reindex(x.index, fill_value=False) & x.notna()
    med = x.rolling(61, center=True, min_periods=10).median(); mad = (x - med).abs().rolling(61, center=True, min_periods=10).median()
    return oor | stuck | (((x - med).abs() > 6 * 1.4826 * mad) & mad.gt(0)) | x.isna()
def pte5(ae, z, n, seq):
    idx = np.arange(seq, n + 1); X = np.stack([z[i - seq:i] for i in idx])[..., None].astype("float32")
    se = (ae.predict(X, verbose=0) - X) ** 2; es, c = np.zeros(n), np.zeros(n)
    for k, end in enumerate(idx - 1): es[end - seq + 1:end + 1] += se[k, :, 0]; c[end - seq + 1:end + 1] += 1
    return es / np.maximum(c, 1)
def build(n):
    inp = tf.keras.Input((SEQ, n)); out = tf.keras.layers.Dense(1)(tf.keras.layers.LSTM(32)(inp)); m = tf.keras.Model(inp, out); m.compile("adam", "mse"); return m

def prepare(split_time, test_end=None):
    """Everything fitted on grid rows before split_time (training partition); returns variants, masks, split index."""
    gtr = grid[grid.index < split_time]; gte = grid[(grid.index >= split_time) & ((grid.index < test_end) if test_end is not None else True)]
    mtr = pd.DataFrame({s: rule_mask5(gtr[s], s) for s in FEATURES}); mte = pd.DataFrame({s: rule_mask5(gte[s], s) for s in FEATURES})
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(42); ae_mask = {}
    for s in FEATURES:
        xc = gtr[s].where(~mtr[s]); mu, sd = xc.mean(), xc.std(); ztr = ((xc - mu) / sd).values
        seqs, starts = [], []
        for i in range(AE_SEQ, len(ztr) + 1):
            w = ztr[i - AE_SEQ:i]
            if not np.isnan(w).any(): seqs.append(w); starts.append(i - AE_SEQ)
        Xtr = np.array(seqs, "float32")[..., None]
        inp = tf.keras.Input((AE_SEQ, 1)); h = tf.keras.layers.LSTM(32)(inp); h = tf.keras.layers.RepeatVector(AE_SEQ)(h)
        out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(tf.keras.layers.LSTM(32, return_sequences=True)(h))
        ae = tf.keras.Model(inp, out); ae.compile("adam", "mse"); ae.fit(Xtr, Xtr, epochs=20, batch_size=256, verbose=0, shuffle=True)
        se = (ae.predict(Xtr, verbose=0) - Xtr) ** 2; es, c = np.zeros(len(ztr)), np.zeros(len(ztr))
        for k, st in enumerate(starts): es[st:st + AE_SEQ] += se[k, :, 0]; c[st:st + AE_SEQ] += 1
        thr = float(np.quantile(es[c > 0] / c[c > 0], 0.995))
        err = pte5(ae, (((gtr[s] - mu) / sd).ffill().bfill()).values, len(gtr), AE_SEQ)
        ae_mask[s] = pd.Series((err > thr) | gtr[s].isna().values, index=gtr.index)
    ae_mask = pd.DataFrame(ae_mask)
    profiles, offsets = {}, {}
    for s in FEATURES:
        xc = gtr[s].where(~mtr[s]); prof = xc.groupby(xc.index.hour * 60 + xc.index.minute).mean().reindex(range(0, 1440, 5)).interpolate(limit_direction="both")
        profiles[s] = prof; pf = pd.Series(prof.reindex(xc.index.hour * 60 + xc.index.minute).values, index=xc.index); offsets[s] = float((xc - pf).mean())
    def repair(mask, s):
        y = gtr[s].where(~mask); gid = y.isna().ne(y.isna().shift()).cumsum(); gl = y.isna().groupby(gid).transform("sum").where(y.isna(), 0)
        lin = y.interpolate("linear", limit_direction="both")
        pf = pd.Series(profiles[s].reindex(y.index.hour * 60 + y.index.minute).values, index=y.index) + offsets[s]
        return lin.where(gl <= LINEAR_MAX, y.fillna(pf))
    variants = {"raw": gtr.ffill().bfill(), "rules_repair": pd.DataFrame({s: repair(mtr[s], s) for s in FEATURES}),
                "ae_repair": pd.DataFrame({s: repair(ae_mask[s], s) for s in FEATURES}), "union_repair": pd.DataFrame({s: repair(mtr[s] | ae_mask[s], s) for s in FEATURES})}
    return gtr, gte, mtr, mte, variants

def run_block(name, split_time, test_end, seeds=SEEDS):
    gtr, gte, mtr, mte, variants = prepare(split_time, test_end)
    split = len(gtr); gall = pd.concat([gtr, gte]); v_all = gall.values.astype("float32"); tcol = gall.columns.get_loc(TARGET)
    ok_row = ~pd.concat([mtr, mte]).any(axis=1).values; flags_tr_row = mtr.any(axis=1).to_numpy()
    y_all = gall[TARGET].to_numpy(dtype="float32")
    ends = np.arange(SEQ, len(gall)); ok = np.array([ok_row[i - SEQ:i + 1].all() for i in ends]); tidx = ends[(ends >= split) & ok]
    # operational test set: every observed target in the test block; inputs forward-filled (then back-filled) within the block, flagged inputs kept as recorded
    obs = np.array([np.isfinite(y_all[i]) for i in ends]); oidx = ends[(ends >= split) & obs]
    # inputs as recorded; missing slots carry the last earlier observation forward (never a later one)
    gtr_ff = gtr.ffill().bfill()
    gte_ff = pd.concat([gtr_ff.tail(1), gte]).ffill().iloc[1:]
    v_op = pd.concat([gtr_ff, gte_ff]).values.astype("float32")
    res = []
    sets = {"screened": (tidx, np.stack([v_all[i - SEQ:i] for i in tidx]) if len(tidx) else None),
            "operational": (oidx, np.stack([v_op[i - SEQ:i] for i in oidx]) if len(oidx) else None)}
    for setname, (idx, Xte) in sets.items():
        if idx is None or len(idx) < 20: continue
        yte = y_all[idx]; pers = y_all[idx - 1]; pers = np.where(np.isfinite(pers), pers, v_op[idx - 1, tcol])
        inside = (yte >= 0) & (yte <= float(gtr[TARGET][~mtr[TARGET]].max()))
        res.append(dict(block=name, test_set=setname, variant="persistence", seed=0, n_test=len(idx), rmse=float(np.sqrt(np.mean((pers - yte) ** 2))), mae=float(np.mean(np.abs(pers - yte))),
                        inside_pct=round(100 * float(inside.mean()), 1), inside_rmse=float(np.sqrt(np.mean((pers - yte)[inside] ** 2))) if inside.any() else np.nan))
        rmse_p = res[-1]["rmse"]
        for vname in ["raw", "rules_repair", "ae_repair", "union_repair", "drop_faulty"]:
            df = variants["raw"] if vname == "drop_faulty" else variants[vname]; v = df.values.astype("float32")
            tr = np.arange(SEQ, split)
            if vname == "drop_faulty": tr = tr[[not flags_tr_row[i - SEQ:i + 1].any() for i in tr]]
            if len(tr) < 20: continue
            Xtr = np.stack([v[i - SEQ:i] for i in tr]); ytr = v[tr, tcol]
            mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-8; ymu, ysd = ytr.mean(), ytr.std()
            for seed in seeds:
                tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed); m = build(3)
                m.fit((Xtr - mu) / sd, (ytr - ymu) / ysd, epochs=EPOCHS, batch_size=BATCH, verbose=0, validation_split=0.1, shuffle=True,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
                p = m.predict((Xte - mu) / sd, verbose=0).ravel() * ysd + ymu
                res.append(dict(block=name, test_set=setname, variant=vname, seed=seed, n_test=len(idx), n_train=len(tr), rmse=float(np.sqrt(np.mean((p - yte) ** 2))), mae=float(np.mean(np.abs(p - yte))),
                                skill=1 - float(np.sqrt(np.mean((p - yte) ** 2))) / rmse_p, inside_pct=round(100 * float(inside.mean()), 1),
                                inside_rmse=float(np.sqrt(np.mean((p - yte)[inside] ** 2))) if inside.any() else np.nan))
        print(name, setname, pd.DataFrame([r for r in res if r["test_set"] == setname and r["block"] == name]).groupby("variant").rmse.mean().round(3).to_dict())
    return res

BLOCKS = [("B1: 5--8 May", pd.Timestamp("2022-05-05"), pd.Timestamp("2022-05-08")),
          ("B2: 8--11 May", pd.Timestamp("2022-05-08"), pd.Timestamp("2022-05-11 01:30")),
          ("B3: 12--22 May (chronological split)", pd.Timestamp("2022-05-12 13:00"), None)]
all_res = []
for name, a, b in BLOCKS: all_res += run_block(name, a, b)
ro = pd.DataFrame(all_res); ro.to_csv(OUT_T / "downstream_rolling_origin_raw.csv", index=False)
summ = ro.groupby(["test_set", "block", "variant"]).agg(n_test=("n_test", "first"), rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), mae=("mae", "mean"),
                                                        skill=("skill", "mean"), inside_pct=("inside_pct", "first"), inside_rmse=("inside_rmse", "mean")).round(3)
summ.to_csv(OUT_T / "downstream_rolling_origin_summary.csv"); print(summ.to_string())
# inference across blocks (screened sets): paired differences of block-level RMSE, n = 3 blocks; sign consistency + paired t
inf_rows = []
piv = ro[ro.test_set == "screened"].groupby(["block", "variant"]).rmse.mean().unstack("variant")
for a, b in [("rules_repair", "raw"), ("ae_repair", "raw"), ("rules_repair", "persistence"), ("ae_repair", "persistence"), ("drop_faulty", "raw"), ("ae_repair", "rules_repair")]:
    if a in piv and b in piv:
        d = (piv[a] - piv[b]).dropna(); t = stats.ttest_rel(piv.loc[d.index, a], piv.loc[d.index, b]) if len(d) > 1 else None
        inf_rows.append({"a": a, "b": b, "n_blocks": int(len(d)), "mean_diff": round(float(d.mean()), 3), "min_diff": round(float(d.min()), 3), "max_diff": round(float(d.max()), 3),
                         "blocks_a_better": int((d < 0).sum()), "paired_t_p": (round(float(t.pvalue), 3) if t is not None else np.nan)})
pd.DataFrame(inf_rows).to_csv(OUT_T / "downstream_rolling_origin_inference.csv", index=False); print(pd.DataFrame(inf_rows).to_string())
print("done")
