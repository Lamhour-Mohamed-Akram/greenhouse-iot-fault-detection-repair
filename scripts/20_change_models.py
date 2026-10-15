# Script 20 - Level-free change models with validation-based selection (protocol: results/change_models/PROTOCOL.md)
#
# Same flags, repairs, training variants, chronological split and common test sets as script 10. For each variant
# and horizon three candidates (persistence, ridge regression on level-free features, LSTM with the soil-moisture
# input expressed relative to its last value) are compared on a chronological validation slice of the training
# period; the selected candidate is refitted on the full training partition and evaluated once on the test set.
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


# ---------------------------------------------------------------- level-free change models
import time
OUT = RES / "change_models"; PRED = OUT / "predictions"; PRED.mkdir(parents=True, exist_ok=True)
ALPHAS = [0.1, 1, 10, 100, 1000]; VAL_SEEDS = [42, 43, 44]
hours = (grid.index.hour + grid.index.minute / 60.0).to_numpy()
def feats(v, i):
    s = v[:, tcol]; h, t = v[:, 0], v[:, 1]
    return np.array([s[i-1]-s[i-2], s[i-1]-s[i-3], s[i-1]-s[i-4], s[i-1]-s[i-7], s[i-1]-s[i-12],
                     h[i-1], t[i-1], h[i-1]-h[i-12], t[i-1]-t[i-12],
                     np.sin(2*np.pi*hours[i-1]/24), np.cos(2*np.pi*hours[i-1]/24)], dtype="float64")
def ridge_fit(F, d, alpha):
    mu, sd = F.mean(0), F.std(0) + 1e-9; Z = (F - mu) / sd; dm = d.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (d - dm))
    return lambda G: ((G - mu) / sd) @ w + dm
def lvlfree(X):
    X = X.copy(); X[:, :, tcol] = X[:, :, tcol] - X[:, -1:, tcol]; return X
def lstm_fit_predict(Xtr, dtr, Xev_list, seeds, tag):
    mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-8; dmu, dsd = dtr.mean(), dtr.std() + 1e-8
    outs = [[] for _ in Xev_list]
    for seed in seeds:
        f = PRED / f"{tag}_seed{seed}.npz"
        if f.exists():
            z = np.load(f); [outs[k].append(z[f"p{k}"]) for k in range(len(Xev_list))]; continue
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed); m = build(len(FEATURES))
        m.fit((Xtr - mu) / sd, (dtr - dmu) / dsd, epochs=EPOCHS, batch_size=BATCH, verbose=0, validation_split=0.1, shuffle=True,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
        ps = [m.predict((X - mu) / sd, verbose=0).ravel() * dsd + dmu for X in Xev_list]
        np.savez_compressed(f, **{f"p{k}": p for k, p in enumerate(ps)}); [outs[k].append(ps[k]) for k in range(len(ps))]
    return outs
def runboot(la, lb, t):
    t = pd.DatetimeIndex(t); br = np.r_[0, np.where(np.diff(t.values).astype("timedelta64[m]").astype(int) > 5)[0] + 1, len(t)]
    segs = [np.arange(br[i], br[i + 1]) for i in range(len(br) - 1)]; o = np.empty(N_BOOT)
    for k in range(N_BOOT):
        sel = np.concatenate([segs[j] for j in RNG.integers(0, len(segs), len(segs))]); o[k] = np.sqrt(la[sel].mean()) - np.sqrt(lb[sel].mean())
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


_PART = {}
def partition_preprocess(cut):
    """Rule masks, autoencoder, profiles, offsets and repaired training variants learned on grid rows [0, cut)
    only (the validation fit partition), plus the rule mask of the validation slice [cut, split) computed on
    that slice only."""
    if cut in _PART: return _PART[cut]
    gp = grid.iloc[:cut]; gv = grid.iloc[cut:split]
    mp = pd.DataFrame({s: rule_mask(gp[s], s) for s in FEATURES}); mv = pd.DataFrame({s: rule_mask(gv[s], s) for s in FEATURES})
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(42); aem = {}
    for s in FEATURES:
        xc = gp[s].where(~mp[s]); mu, sd = xc.mean(), xc.std(); z = ((xc - mu) / sd).values
        seqs, starts = [], []
        for i in range(AE_SEQ, len(z) + 1):
            w = z[i - AE_SEQ:i]
            if not np.isnan(w).any(): seqs.append(w); starts.append(i - AE_SEQ)
        Xa = np.array(seqs, "float32")[..., None]
        inp = tf.keras.Input((AE_SEQ, 1)); h = tf.keras.layers.LSTM(32)(inp); h = tf.keras.layers.RepeatVector(AE_SEQ)(h)
        out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(tf.keras.layers.LSTM(32, return_sequences=True)(h))
        ae = tf.keras.Model(inp, out); ae.compile("adam", "mse"); ae.fit(Xa, Xa, epochs=20, batch_size=256, verbose=0, shuffle=True)
        se = (ae.predict(Xa, verbose=0) - Xa) ** 2; es, c = np.zeros(len(z)), np.zeros(len(z))
        for k, st in enumerate(starts): es[st:st + AE_SEQ] += se[k, :, 0]; c[st:st + AE_SEQ] += 1
        thr = float(np.quantile(es[c > 0] / c[c > 0], 0.995))
        err = per_timestep_error(ae, (((gp[s] - mu) / sd).ffill().bfill()).values, len(gp), AE_SEQ)
        aem[s] = pd.Series((err > thr) | gp[s].isna().values, index=gp.index)
    aem = pd.DataFrame(aem)
    prof_p, off_p = {}, {}
    for s in FEATURES:
        xc = gp[s].where(~mp[s])
        pr = xc.groupby(xc.index.hour * 60 + xc.index.minute).mean().reindex(range(0, 1440, 5)).interpolate(limit_direction="both")
        prof_p[s] = pr; pf = pd.Series(pr.reindex(xc.index.hour * 60 + xc.index.minute).values, index=xc.index); off_p[s] = float((xc - pf).mean())
    def rep_p(mask, s):
        y = gp[s].where(~mask); gid = y.isna().ne(y.isna().shift()).cumsum(); gl = y.isna().groupby(gid).transform("sum").where(y.isna(), 0)
        lin = y.interpolate("linear", limit_direction="both")
        pf = pd.Series(prof_p[s].reindex(y.index.hour * 60 + y.index.minute).values, index=y.index) + off_p[s]
        return lin.where(gl <= LINEAR_MAX, y.fillna(pf))
    var_p = {"raw": gp.ffill().bfill(), "rules_repair": pd.DataFrame({s: rep_p(mp[s], s) for s in FEATURES}),
             "ae_repair": pd.DataFrame({s: rep_p(aem[s], s) for s in FEATURES}), "union_repair": pd.DataFrame({s: rep_p(mp[s] | aem[s], s) for s in FEATURES})}
    _PART[cut] = (var_p, mp.any(axis=1).to_numpy(), ~mv.any(axis=1).to_numpy())
    return _PART[cut]

VARIANTS = ["raw", "rules_repair", "ae_repair", "union_repair", "drop_faulty"]
sel_rows, test_rows, inf_rows = [], [], []
for H in HORIZONS:
    tidx = test_index(H); yte, base, _ = baselines(tidx, H); pers = base["persistence"]
    inside = (yte >= SUPPORT[0]) & (yte <= SUPPORT[1]); ts = grid.index[tidx + H - 1]
    Xte_raw = np.stack([v_orig[i - SEQ:i] for i in tidx]); Fte = np.stack([feats(v_orig, i) for i in tidx]); last_te = v_orig[tidx - 1, tcol]
    losses = {"persistence": (pers - yte) ** 2}
    for name in VARIANTS:
        df = variants["raw"] if name == "drop_faulty" else variants[name]; v = df.values.astype("float32")
        idx = np.arange(SEQ, split - H + 1)
        if name == "drop_faulty": idx = idx[[not flags_tr_row[i - SEQ:i + H].any() for i in idx]]
        cut = int(np.arange(SEQ, split - H + 1)[int(0.8 * len(np.arange(SEQ, split - H + 1)))])
        var_p, flags_p, ok_val = partition_preprocess(cut)
        vp = (var_p["raw"] if name == "drop_faulty" else var_p[name]).values.astype("float32")
        fit_idx = np.arange(SEQ, cut - H + 1)
        if name == "drop_faulty": fit_idx = fit_idx[[not flags_p[i - SEQ:i + H].any() for i in fit_idx]]
        val_idx = np.arange(cut + SEQ, split - H + 1)
        val_idx = val_idx[[ok_val[i - SEQ - cut:i - cut].all() and ok_val[i + H - 1 - cut] for i in val_idx]]
        d_of = lambda vv, ii: vv[ii + H - 1, tcol] - vv[ii - 1, tcol]
        Ffit = np.stack([feats(vp, i) for i in fit_idx]); dfit = d_of(vp, fit_idx)
        Fval = np.stack([feats(v_orig, i) for i in val_idx]); dval = d_of(v_orig, val_idx)
        # candidate validation RMSE
        val = {"persistence": float(np.sqrt(np.mean(dval ** 2)))}
        ridge_val = {a: float(np.sqrt(np.mean((ridge_fit(Ffit, dfit, a)(Fval) - dval) ** 2))) for a in ALPHAS}
        alpha = min(ridge_val, key=ridge_val.get); val["ridge"] = ridge_val[alpha]
        Xfit = lvlfree(np.stack([vp[i - SEQ:i] for i in fit_idx])); Xval = lvlfree(np.stack([v_orig[i - SEQ:i] for i in val_idx]))
        pv = lstm_fit_predict(Xfit, dfit, [Xval], VAL_SEEDS, f"val2_H{H}_{name}")[0]
        val["lstm"] = float(np.sqrt(np.mean(np.mean([(p - dval) ** 2 for p in pv], axis=0))))
        order = ["persistence", "ridge", "lstm"]; best = min(order, key=lambda k: (round(val[k], 6), order.index(k)))
        sel_rows.append({"H": H, "variant": name, "n_val": int(len(val_idx)), "val_rmse_persistence": val["persistence"], "val_rmse_ridge": val["ridge"],
                         "ridge_alpha": alpha, "val_rmse_lstm": val["lstm"], "selected": best})
        # full refit and single test evaluation of every candidate
        Fall = np.stack([feats(v, i) for i in idx]); dall = d_of(v, idx)
        p_ridge = last_te + ridge_fit(Fall, dall, alpha)(Fte)
        Xall = lvlfree(np.stack([v[i - SEQ:i] for i in idx]))
        p_lstm = [last_te + p for p in lstm_fit_predict(Xall, dall, [lvlfree(Xte_raw)], SEEDS, f"test_H{H}_{name}")[0]]
        np.savez_compressed(PRED / f"ridge_H{H}_{name}.npz", pred=p_ridge)
        cands = {"ridge": [p_ridge], "lstm": p_lstm, "persistence": [pers]}
        for cname, plist in cands.items():
            for k, p in enumerate(plist):
                e = p - yte
                test_rows.append({"H": H, "variant": name, "candidate": cname, "selected": cname == best, "seed": SEEDS[k] if cname == "lstm" else "",
                                  "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
                                  "rmse_inside": float(np.sqrt(np.mean(e[inside] ** 2))), "rmse_outside": float(np.sqrt(np.mean(e[~inside] ** 2)))})
        losses[f"{name}|ridge"] = (p_ridge - yte) ** 2; losses[f"{name}|lstm"] = ((np.array(p_lstm) - yte) ** 2).mean(axis=0)
        losses[f"{name}|selected"] = losses["persistence"] if best == "persistence" else losses[f"{name}|{best}"]
        print(time.strftime("%H:%M:%S"), H, name, "selected", best, {k: round(x, 4) for k, x in val.items()}, flush=True)
    pairs = [(f"{v}|{c}", "persistence") for v in VARIANTS for c in ["selected", "ridge", "lstm"]] + \
            [(f"{v}|selected", "raw|selected") for v in VARIANTS[1:]] + [(f"{v}|ridge", "raw|ridge") for v in VARIANTS[1:]]
    for scope, sm in [("full", np.ones(len(yte), bool)), ("inside_support", inside), ("outside_support", ~inside)]:
        for a, b in pairs:
            la, lb = losses[a][sm], losses[b][sm]
            if np.allclose(la, lb):
                inf_rows.append({"H": H, "scope": scope, "a": a, "b": b, "n": int(sm.sum()), "pooled_rmse_a": float(np.sqrt(la.mean())), "pooled_rmse_b": float(np.sqrt(lb.mean())), "diff": 0.0, "note": "identical predictions"}); continue
            lo, hi = block_bootstrap_ci(la, lb); rlo, rhi = runboot(la, lb, ts[sm]); dmv, pv_ = dm_test(la, lb, H)
            inf_rows.append({"H": H, "scope": scope, "a": a, "b": b, "n": int(sm.sum()), "pooled_rmse_a": float(np.sqrt(la.mean())), "pooled_rmse_b": float(np.sqrt(lb.mean())),
                             "diff": float(np.sqrt(la.mean()) - np.sqrt(lb.mean())), "b24_lo": lo, "b24_hi": hi, "run_lo": rlo, "run_hi": rhi, "dm": dmv, "p": pv_})
pd.DataFrame(sel_rows).round(5).to_csv(OUT / "selection.csv", index=False)
tr = pd.DataFrame(test_rows); tr.to_csv(OUT / "test_metrics_all_candidates.csv", index=False)
pd.DataFrame(inf_rows).round(4).to_csv(OUT / "temporal_inference.csv", index=False)
print(pd.DataFrame(sel_rows).round(4).to_string())
print(tr.groupby(["H", "variant", "candidate", "selected"])[["rmse", "rmse_inside", "rmse_outside"]].mean().round(3).to_string())
print("done")
