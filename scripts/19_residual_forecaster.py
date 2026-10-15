# Script 19 - Residual (change) LSTM forecaster (protocol: results/residual_forecaster/PROTOCOL.md)
#
# Same flags, repairs, training variants, chronological split and common test sets as script 10; the LSTM is
# trained on the change y(t+H) - y(t) instead of the level, and the prediction is the last observed value plus
# the predicted change. The absolute level is still among the inputs.
import json, platform, sys
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from scipy import stats

tf.config.experimental.enable_op_determinism()
ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
DATA = ROOT / "data" / "data.csv.gz"
RES = ROOT / "results"; OUT_T = RES / "tables"; OUT_F = RES / "figures"
OUT_T.mkdir(parents=True, exist_ok=True); OUT_F.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(42, 52)); TARGET = "soil_moisture"
FEATURES = ["humidity", "temperature", "soil_moisture"]
FREQ = "5min"; SEQ = 12; HORIZONS = [1, 12, 72]
VALID = {"humidity": (0, 100), "temperature": (-5, 55), "soil_moisture": (0, 100)}
LINEAR_MAX = 48; AE_SEQ = 12; EPOCHS, BATCH = 30, 256
BLOCK = 24; N_BOOT = 2000; RNG = np.random.default_rng(0)

raw = pd.read_csv(DATA, encoding="latin-1")
raw.columns = ["Id", "humidity", "temperature", "soil_moisture", "soil_temp", "co2", "light", "date"]
raw["date"] = pd.to_datetime(raw["date"]); raw = raw.sort_values("date").reset_index(drop=True)
grid = raw.set_index("date")[FEATURES].resample(FREQ).median()
split = int(0.7 * len(grid)); gtr, gte = grid.iloc[:split], grid.iloc[split:]
print(f"grid {len(grid)} rows, split at {grid.index[split]}, train {split}, test {len(gte)}")

# ---------------------------------------------------------------- flags
def raw_stuck_to_grid(s):
    x = raw[s]; d = raw["date"]; gap = d.diff().dt.total_seconds() > 300
    g = ((x.diff() != 0) | gap).cumsum()
    st = d.groupby(g).transform("first"); en = d.groupby(g).transform("last")
    dur = (en - st).dt.total_seconds()
    m = pd.Series((dur > 3600).values, index=d)
    return m.groupby(pd.Grouper(freq=FREQ)).any().reindex(grid.index, fill_value=False)
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


def rule_mask(x, s, mode="raw"):
    lo, hi = VALID[s]; oor = ((x < lo) | (x > hi)) & x.notna()
    if mode == "raw":
        stuck = raw_stuck_window(s, x.index[0], x.index[-1] + pd.Timedelta(FREQ), FREQ).reindex(x.index, fill_value=False) & x.notna()
    else:
        grp = (x.diff() != 0).cumsum(); runlen = x.groupby(grp).transform("size")
        stuck = (runlen > 12) & x.notna()
    med = x.rolling(61, center=True, min_periods=10).median()
    mad = (x - med).abs().rolling(61, center=True, min_periods=10).median()
    spike = ((x - med).abs() > 6 * 1.4826 * mad) & mad.gt(0)
    return oor | stuck | spike | x.isna()

mtr = pd.DataFrame({s: rule_mask(gtr[s], s) for s in FEATURES})
mte = pd.DataFrame({s: rule_mask(gte[s], s) for s in FEATURES})
masks = pd.concat([mtr, mte])
mtr_grid = pd.DataFrame({s: rule_mask(gtr[s], s, "grid") for s in FEATURES})
flag_stats = {
    "rules_raw_stuck_train_pct": {s: round(100 * float(mtr[s].mean()), 1) for s in FEATURES},
    "rules_grid_stuck_train_pct": {s: round(100 * float(mtr_grid[s].mean()), 1) for s in FEATURES},
    "row_any_fault_pct_raw_stuck": round(100 * float(masks.any(axis=1).mean()), 1),
    "row_any_fault_pct_grid_stuck_train": round(100 * float(mtr_grid.any(axis=1).mean()), 1),
}
print(flag_stats)

def clean_tr(s): return gtr[s].where(~mtr[s])

def per_timestep_error(ae, z, n, seq):
    idx = np.arange(seq, n + 1)
    X = np.stack([z[i - seq:i] for i in idx])[..., None].astype("float32")
    se = (ae.predict(X, verbose=0) - X) ** 2
    es, c = np.zeros(n), np.zeros(n)
    for k, end in enumerate(idx - 1):
        es[end - seq + 1:end + 1] += se[k, :, 0]; c[end - seq + 1:end + 1] += 1
    return es / np.maximum(c, 1)

tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(42)
ae_mask = {}
for s in FEATURES:
    xc = clean_tr(s); mu, sd = xc.mean(), xc.std(); ztr = ((xc - mu) / sd).values
    seqs, starts = [], []
    for i in range(AE_SEQ, len(ztr) + 1):
        w = ztr[i - AE_SEQ:i]
        if not np.isnan(w).any(): seqs.append(w); starts.append(i - AE_SEQ)
    Xtr = np.array(seqs, "float32")[..., None]
    inp = tf.keras.Input((AE_SEQ, 1)); h = tf.keras.layers.LSTM(32)(inp); h = tf.keras.layers.RepeatVector(AE_SEQ)(h)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(tf.keras.layers.LSTM(32, return_sequences=True)(h))
    ae = tf.keras.Model(inp, out); ae.compile("adam", "mse")
    ae.fit(Xtr, Xtr, epochs=20, batch_size=256, verbose=0, shuffle=True)
    se = (ae.predict(Xtr, verbose=0) - Xtr) ** 2; es, c = np.zeros(len(ztr)), np.zeros(len(ztr))
    for k, st in enumerate(starts): es[st:st + AE_SEQ] += se[k, :, 0]; c[st:st + AE_SEQ] += 1
    thr = float(np.quantile(es[c > 0] / c[c > 0], 0.995))
    zf = (((gtr[s] - mu) / sd).ffill().bfill()).values
    err = per_timestep_error(ae, zf, len(zf), AE_SEQ)
    ae_mask[s] = pd.Series((err > thr) | gtr[s].isna().values, index=gtr.index)
    print(f"AE {s}: {len(Xtr)} clean seqs, thr={thr:.4f}, train mask {100 * ae_mask[s].mean():.1f}%")
ae_mask = pd.DataFrame(ae_mask); union_mask = mtr | ae_mask
flag_stats["ae_train_pct"] = {s: round(100 * float(ae_mask[s].mean()), 1) for s in FEATURES}
flag_stats["union_train_pct"] = {s: round(100 * float(union_mask[s].mean()), 1) for s in FEATURES}

# ---------------------------------------------------------------- repair
profiles, offsets = {}, {}
for s in FEATURES:
    xc = clean_tr(s)
    prof = xc.groupby(xc.index.hour * 60 + xc.index.minute).mean().reindex(range(0, 1440, 5)).interpolate(limit_direction="both")
    profiles[s] = prof
    pf = pd.Series(prof.reindex(xc.index.hour * 60 + xc.index.minute).values, index=xc.index)
    offsets[s] = float((xc - pf).mean())

def repair(mask, s, long_fill="profile"):
    y = gtr[s].where(~mask)
    gid = y.isna().ne(y.isna().shift()).cumsum()
    gl = y.isna().groupby(gid).transform("sum").where(y.isna(), 0)
    lin = y.interpolate("linear", limit_direction="both")
    if long_fill == "none":
        return lin.where(gl <= LINEAR_MAX, np.nan)
    pf = pd.Series(profiles[s].reindex(y.index.hour * 60 + y.index.minute).values, index=y.index) + offsets[s]
    return lin.where(gl <= LINEAR_MAX, y.fillna(pf))

variants = {"raw": gtr.ffill().bfill(),
            "rules_repair": pd.DataFrame({s: repair(mtr[s], s) for s in FEATURES}),
            "ae_repair": pd.DataFrame({s: repair(ae_mask[s], s) for s in FEATURES}),
            "union_repair": pd.DataFrame({s: repair(union_mask[s], s) for s in FEATURES})}
for k, df in variants.items(): assert np.isfinite(df.values).all(), k

# long-gap bookkeeping
gaps_raw = raw["date"].diff().dt.total_seconds()
gi = np.where(gaps_raw > 300)[0]
gaps = pd.DataFrame({"start": raw["date"].iloc[gi - 1].values, "end": raw["date"].iloc[gi].values,
                     "h": gaps_raw.iloc[gi].values / 3600})
long_gaps = gaps[gaps.h > 12]
in_long = pd.Series(False, index=gtr.index)
for _, g in long_gaps.iterrows(): in_long |= (gtr.index > g.start) & (gtr.index < g.end)
profile_rows = {}
for name, m in [("rules", mtr[TARGET]), ("ae", ae_mask[TARGET]), ("union", union_mask[TARGET])]:
    gid = m.ne(m.shift()).cumsum(); runs = m.groupby(gid).agg(["first", "size"]); runs = runs[runs["first"]]
    profile_rows[name] = int(runs.loc[runs["size"] > LINEAR_MAX, "size"].sum())
longgap_stats = {"train_rows": int(len(gtr)), "train_rows_in_gt12h_outage": int(in_long.sum()),
                 "train_rows_in_gt12h_outage_pct": round(100 * float(in_long.mean()), 1),
                 "train_target_rows_profile_filled_gt4h": profile_rows,
                 "train_target_rows_profile_filled_gt4h_pct": {k: round(100 * v / len(gtr), 1) for k, v in profile_rows.items()}}
print(longgap_stats)

# ---------------------------------------------------------------- test sets, baselines
ok_row = ~masks.any(axis=1).values
v_orig = grid.values.astype("float32"); tcol = grid.columns.get_loc(TARGET)
y_all = grid[TARGET].to_numpy(dtype="float32")
train_clean_targets = gtr[TARGET][~mtr[TARGET]].dropna()
SUPPORT = (float(train_clean_targets.min()), float(train_clean_targets.max()))
flags_tr_row = mtr.any(axis=1).to_numpy()

def test_index(H):
    ends = np.arange(SEQ, len(grid) - H + 1)
    ok = np.array([ok_row[i - SEQ:i].all() and ok_row[i + H - 1] for i in ends])
    return ends[(ends >= split) & ok]

def baselines(tidx, H):
    yt = y_all[tidx + H - 1]; pers = y_all[tidx - 1]
    # seasonal naive: value observed 24 h before the target, else persistence
    j = tidx + H - 1 - 288; seas = np.where((j >= 0), y_all[np.clip(j, 0, None)], np.nan)
    seas = np.where(np.isfinite(seas), seas, pers)
    # AR(1) on clean training pairs (x_{t-1} -> y_{t+H-1})
    tr = np.arange(SEQ, split - H + 1); keep = np.array([ok_row[i - 1] and ok_row[i + H - 1] for i in tr]); tr = tr[keep]
    A = np.c_[np.ones(len(tr)), y_all[tr - 1]]; coef = np.linalg.lstsq(A, y_all[tr + H - 1], rcond=None)[0]
    ar1 = coef[0] + coef[1] * pers
    return yt, {"persistence": pers, "seasonal_naive": seas, "ar1": ar1}, coef

def metrics(p, y):
    e = p - y
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)))

def build(n):
    inp = tf.keras.Input((SEQ, n)); out = tf.keras.layers.Dense(1)(tf.keras.layers.LSTM(32)(inp))
    m = tf.keras.Model(inp, out); m.compile("adam", "mse"); return m

def train_eval(df, name, H, tidx, Xte, yte, seeds=SEEDS, window_filter=None):
    v = df.values.astype("float32"); idx = np.arange(SEQ, split - H + 1)
    if name == "drop_faulty":
        idx = idx[[not flags_tr_row[i - SEQ:i + H].any() for i in idx]]
    if window_filter is not None:
        idx = idx[[window_filter(i) for i in idx]]
    if len(idx) < 20:
        return [], idx
    Xtr = np.stack([v[i - SEQ:i] for i in idx]); ytr = v[idx + H - 1, tcol]
    fin = np.isfinite(Xtr).all(axis=(1, 2)) & np.isfinite(ytr); Xtr, ytr, idx = Xtr[fin], ytr[fin], idx[fin]
    if len(idx) < 20:
        return [], idx
    mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-8; ymu, ysd = ytr.mean(), ytr.std()
    Xs, ys, Xts = (Xtr - mu) / sd, (ytr - ymu) / ysd, (Xte - mu) / sd
    preds = []
    for seed in seeds:
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed); m = build(len(FEATURES))
        m.fit(Xs, ys, epochs=EPOCHS, batch_size=BATCH, verbose=0, validation_split=0.1, shuffle=True,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
        preds.append(m.predict(Xts, verbose=0).ravel() * ysd + ymu)
    return preds, idx

# ---------------------------------------------------------------- block bootstrap + DM
def block_bootstrap_ci(l_a, l_b, block=BLOCK, n_boot=N_BOOT):
    """95% CI of RMSE_a - RMSE_b from moving-block bootstrap of paired per-sample squared losses."""
    n = len(l_a); nb = int(np.ceil(n / block)); diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = RNG.integers(0, n - block + 1, nb)
        sel = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        diffs[b] = np.sqrt(l_a[sel].mean()) - np.sqrt(l_b[sel].mean())
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

def dm_test(l_a, l_b, h):
    """Diebold-Mariano test on loss differential with Bartlett HAC (lag h+SEQ-1) and Harvey correction."""
    d = l_a - l_b; n = len(d); dbar = d.mean(); d0 = d - dbar
    L = h + SEQ - 1; gamma0 = np.mean(d0 * d0); var = gamma0
    for k in range(1, L + 1):
        gk = np.mean(d0[k:] * d0[:-k]); var += 2 * (1 - k / (L + 1)) * gk
    var = max(var, 1e-12)
    dm = dbar / np.sqrt(var / n)
    harvey = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_h = dm * harvey
    p = 2 * (1 - stats.t.cdf(abs(dm_h), df=n - 1))
    return float(dm_h), float(p)


# ---------------------------------------------------------------- residual forecaster (only the target changes)
import time
OUT = RES / "residual_forecaster"; PRED = OUT / "predictions"; PRED.mkdir(parents=True, exist_ok=True)
def run_resid(df, name, H, Xte, last_obs):
    v = df.values.astype("float32"); idx = np.arange(SEQ, split - H + 1)
    if name == "drop_faulty":
        idx = idx[[not flags_tr_row[i - SEQ:i + H].any() for i in idx]]
    Xtr = np.stack([v[i - SEQ:i] for i in idx]); dtr = v[idx + H - 1, tcol] - v[idx - 1, tcol]
    fin = np.isfinite(Xtr).all(axis=(1, 2)) & np.isfinite(dtr); Xtr, dtr, idx = Xtr[fin], dtr[fin], idx[fin]
    mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-8; dmu, dsd = dtr.mean(), dtr.std() + 1e-8
    Xs, ds, Xts = (Xtr - mu) / sd, (dtr - dmu) / dsd, (Xte - mu) / sd
    out = []
    for seed in SEEDS:
        f = PRED / f"H{H}_{name}_seed{seed}.npz"
        if f.exists(): out.append(np.load(f)["pred"]); continue
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed); m = build(len(FEATURES))
        m.fit(Xs, ds, epochs=EPOCHS, batch_size=BATCH, verbose=0, validation_split=0.1, shuffle=True,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
        p = last_obs + (m.predict(Xts, verbose=0).ravel() * dsd + dmu)
        np.savez_compressed(f, pred=p); out.append(p)
    return out, len(idx)

def runboot(la, lb, t, n_boot=N_BOOT):
    t = pd.DatetimeIndex(t); br = np.r_[0, np.where(np.diff(t.values).astype("timedelta64[m]").astype(int) > 5)[0] + 1, len(t)]
    segs = [np.arange(br[i], br[i + 1]) for i in range(len(br) - 1)]; o = np.empty(n_boot)
    for k in range(n_boot):
        sel = np.concatenate([segs[j] for j in RNG.integers(0, len(segs), len(segs))]); o[k] = np.sqrt(la[sel].mean()) - np.sqrt(lb[sel].mean())
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))

VARIANTS = ["raw", "rules_repair", "ae_repair", "union_repair", "drop_faulty"]
rows, inf_rows = [], []
for H in HORIZONS:
    tidx = test_index(H); yte, base, _ = baselines(tidx, H); pers = base["persistence"]
    Xte = np.stack([v_orig[i - SEQ:i] for i in tidx]); last_obs = v_orig[tidx - 1, tcol]
    assert np.allclose(last_obs, pers)
    inside = (yte >= SUPPORT[0]) & (yte <= SUPPORT[1]); ts = grid.index[tidx + H - 1]
    lvl = np.load(OUT_T / f"downstream_losses_H{H}.npz")
    losses = {"persistence": (pers - yte) ** 2}
    for name in VARIANTS:
        df = variants["raw"] if name == "drop_faulty" else variants[name]
        preds, ntr = run_resid(df, name, H, Xte, last_obs)
        for seed, p in zip(SEEDS, preds):
            e = p - yte
            rows.append({"H": H, "variant": name, "seed": seed, "n_train": ntr, "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
                         "rmse_inside": float(np.sqrt(np.mean(e[inside] ** 2))), "rmse_outside": float(np.sqrt(np.mean(e[~inside] ** 2))) if (~inside).any() else np.nan,
                         "skill": 1 - float(np.sqrt(np.mean(e ** 2))) / float(np.sqrt(np.mean((pers - yte) ** 2)))})
        losses[name] = ((np.array(preds) - yte) ** 2).mean(axis=0)
        print(time.strftime("%H:%M:%S"), H, name, round(float(np.mean([r["rmse"] for r in rows if r["H"] == H and r["variant"] == name])), 3), flush=True)
    pairs = [(v, "persistence") for v in VARIANTS] + [(v, "raw") for v in VARIANTS[1:]] + [("ae_repair", "rules_repair")]
    for scope, sel in [("full", np.ones(len(yte), bool)), ("inside_support", inside), ("outside_support", ~inside)]:
        if sel.sum() < 20: continue
        for a, b in pairs:
            la, lb = losses[a][sel], losses[b][sel]
            lo, hi = block_bootstrap_ci(la, lb); rlo, rhi = runboot(la, lb, ts[sel]); dmv, pv = dm_test(la, lb, H)
            inf_rows.append({"H": H, "scope": scope, "a": a, "b": b, "n": int(sel.sum()), "pooled_rmse_a": float(np.sqrt(la.mean())), "pooled_rmse_b": float(np.sqrt(lb.mean())),
                             "diff": float(np.sqrt(la.mean()) - np.sqrt(lb.mean())), "b24_lo": lo, "b24_hi": hi, "run_lo": rlo, "run_hi": rhi, "dm": dmv, "p": pv})
        for v in VARIANTS:   # residual vs level model of the same variant (seed-averaged losses of script 10)
            if v in lvl.files:
                la, lb = losses[v][sel], lvl[v][sel]
                lo, hi = block_bootstrap_ci(la, lb); rlo, rhi = runboot(la, lb, ts[sel])
                inf_rows.append({"H": H, "scope": scope, "a": f"{v} (residual)", "b": f"{v} (level)", "n": int(sel.sum()), "pooled_rmse_a": float(np.sqrt(la.mean())),
                                 "pooled_rmse_b": float(np.sqrt(lb.mean())), "diff": float(np.sqrt(la.mean()) - np.sqrt(lb.mean())), "b24_lo": lo, "b24_hi": hi, "run_lo": rlo, "run_hi": rhi})
    np.savez_compressed(OUT / f"residual_losses_H{H}.npz", inside=inside, yte=yte, ts=np.array(ts.astype("int64")), **losses)
res = pd.DataFrame(rows); res.to_csv(OUT / "per_fit_metrics.csv", index=False)
summ = res.groupby(["H", "variant"])[["rmse", "mae", "rmse_inside", "rmse_outside", "skill", "n_train"]].agg(["mean", "std"]).round(4); summ.to_csv(OUT / "summary.csv")
inf = pd.DataFrame(inf_rows).round(4); inf.to_csv(OUT / "temporal_inference.csv", index=False)
print(summ.to_string()); print(inf[inf.b.isin(["persistence", "raw"])].to_string()); print("done")
