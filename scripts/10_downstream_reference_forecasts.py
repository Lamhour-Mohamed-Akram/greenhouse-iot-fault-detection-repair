# Script 10 - Downstream evaluation with reference forecasts
#
# Same five training-data variants as script 05b, with these differences:
#   - stuck-at flags on the 5-min grid come from the raw 1-s samples (exact constancy > 1 h,
#     runs split at > 5-min gaps) instead of from constant runs of 5-min medians, which over-flag
#     the slow integer-quantized soil-moisture channel
#   - three forecast horizons (5 min, 1 h, 6 h)
#   - persistence, seasonal-naive and AR(1) references; skill relative to persistence
#     (SS = 1 - RMSE/RMSE_pers, relative MAE = MAE/MAE_pers)
#   - test error split inside / outside the training support
#   - moving-block bootstrap intervals and Diebold-Mariano tests on paired per-sample losses;
#     the seed comparison is kept as a robustness check only
#   - test-set timing and scope statistics
#   - ablation of the treatment of flagged runs longer than 12 h
import json, platform, sys
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from scipy import stats

tf.config.experimental.enable_op_determinism()
ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
DATA = ROOT / "data" / "data.csv.gz"
RES = ROOT / "results"; OUT_T = RES / "tables"; OUT_F = RES / "figures"
TAB, INJ, MODELS, OFIG, ORIG_T, MS_F = OUT_T, RES / "injected", RES / "models", OUT_F, OUT_T, OUT_F
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

# ---------------------------------------------------------------- main experiment
VARIANTS = ["raw", "rules_repair", "ae_repair", "union_repair", "drop_faulty"]
rows, base_rows, infer_rows, decomp_rows, store = [], [], [], [], {}
test_stats = {}
for H in HORIZONS:
    tidx = test_index(H); yte, base, ar_coef = baselines(tidx, H)
    Xte = np.stack([v_orig[i - SEQ:i] for i in tidx]); assert np.isfinite(Xte).all() and np.isfinite(yte).all()
    inside = (yte >= SUPPORT[0]) & (yte <= SUPPORT[1])
    ts = grid.index[tidx + H - 1]
    test_stats[H] = {"n_test": int(len(tidx)), "test_period_rows": int(len(gte)),
                     "retained_pct": round(100 * len(tidx) / len(gte), 1),
                     "first": str(ts.min()), "last": str(ts.max()),
                     "per_day": {str(k): int(v) for k, v in pd.Series(ts.date).value_counts().sort_index().items()},
                     "target_mean": round(float(yte.mean()), 2), "target_sd": round(float(yte.std()), 2),
                     "target_min": float(yte.min()), "target_max": float(yte.max()),
                     "inside_support_n": int(inside.sum()), "inside_support_pct": round(100 * float(inside.mean()), 1),
                     "ar1_coef": [round(float(c), 4) for c in ar_coef]}
    if H == 1:
        excl = gte[TARGET].drop(ts, errors="ignore").dropna()
        test_stats[H]["excluded_observed_n"] = int(len(excl)); test_stats[H]["excluded_mean"] = round(float(excl.mean()), 2)
        test_stats[H]["excluded_sd"] = round(float(excl.std()), 2)
    losses = {}
    for bname, bp in base.items():
        m = metrics(bp, yte); base_rows.append(dict(H=H, variant=bname, **m))
        losses[bname] = (bp - yte) ** 2
        decomp_rows.append(dict(H=H, variant=bname, inside_rmse=float(np.sqrt(np.mean((bp - yte)[inside] ** 2))),
                                outside_rmse=float(np.sqrt(np.mean((bp - yte)[~inside] ** 2))) if (~inside).any() else np.nan))
    rmse_p, mae_p = metrics(base["persistence"], yte)["rmse"], metrics(base["persistence"], yte)["mae"]
    print(f"\n=== H={H} n_test={len(tidx)} persistence RMSE={rmse_p:.3f} inside-support {inside.mean()*100:.1f}%")
    for name in VARIANTS:
        df = variants["raw"] if name == "drop_faulty" else variants[name]
        preds, idx = train_eval(df, name, H, tidx, Xte, yte)
        if not preds:
            rows.append(dict(H=H, variant=name, seed=np.nan, rmse=np.nan, mae=np.nan, r2=np.nan, skill=np.nan, mase=np.nan, n_train=len(idx)))
            print(f" {name}: {len(idx)} windows, skipped"); continue
        for seed, p in zip(SEEDS, preds):
            m = metrics(p, yte)
            rows.append(dict(H=H, variant=name, seed=seed, **m, skill=1 - m["rmse"] / rmse_p, mase=m["mae"] / mae_p, n_train=len(idx),
                             inside_rmse=float(np.sqrt(np.mean((p - yte)[inside] ** 2))),
                             outside_rmse=float(np.sqrt(np.mean((p - yte)[~inside] ** 2))) if (~inside).any() else np.nan))
        P = np.array(preds); losses[name] = ((P - yte) ** 2).mean(axis=0)   # seed-averaged per-sample loss
        store[(H, name)] = (ts, yte, preds[0])
        d = [r for r in rows if r["H"] == H and r["variant"] == name]
        decomp_rows.append(dict(H=H, variant=name, inside_rmse=float(np.mean([r["inside_rmse"] for r in d])),
                                outside_rmse=float(np.nanmean([r["outside_rmse"] for r in d]))))
        print(f" {name:13s} n_train={len(idx):5d} RMSE={np.mean([r['rmse'] for r in d]):.3f}+/-{np.std([r['rmse'] for r in d]):.3f} "
              f"SS={np.mean([r['skill'] for r in d]):+.2f} inside={decomp_rows[-1]['inside_rmse']:.3f} outside={decomp_rows[-1]['outside_rmse']:.3f}")
    # temporal inference: each variant vs persistence and vs raw; ae vs rules
    pairs = [(v, "persistence") for v in VARIANTS if v in losses] + [(v, "raw") for v in VARIANTS[1:] if v in losses] + [("ae_repair", "rules_repair")]
    for a, b in pairs:
        if a not in losses or b not in losses: continue
        for scope, sel in [("all", np.ones(len(yte), bool)), ("inside_support", inside)]:
            la, lb = losses[a][sel], losses[b][sel]
            lo, hi = block_bootstrap_ci(la, lb); dmv, pv = dm_test(la, lb, H)
            infer_rows.append(dict(H=H, scope=scope, a=a, b=b, n=int(sel.sum()), rmse_a=float(np.sqrt(la.mean())), rmse_b=float(np.sqrt(lb.mean())),
                                   diff=float(np.sqrt(la.mean()) - np.sqrt(lb.mean())), ci_lo=lo, ci_hi=hi, dm=dmv, p=pv))
    np.savez(OUT_T / f"downstream_losses_H{H}.npz", inside=inside, yte=yte, ts=np.array(ts.astype("int64")), **{k: v for k, v in losses.items()})
    # paired seed comparison (robustness only)
    if H == 1:
        rr = pd.DataFrame(rows); rr = rr[(rr.H == 1)]
        seedcmp = {}
        for b in ["raw", "rules_repair", "union_repair", "drop_faulty"]:
            x = rr[rr.variant == "ae_repair"].sort_values("seed").rmse.values; y = rr[rr.variant == b].sort_values("seed").rmse.values
            if len(x) == len(y) == 10:
                w = stats.wilcoxon(x, y, alternative="two-sided", method="exact")
                seedcmp[f"ae_vs_{b}"] = {"wins": int((x < y).sum()), "p": float(w.pvalue)}
        test_stats["seed_wilcoxon_H1"] = seedcmp

res = pd.DataFrame(rows); res.to_csv(OUT_T / "downstream_ref_raw.csv", index=False)
summ = res.groupby(["H", "variant"])[["rmse", "mae", "r2", "skill", "mase", "inside_rmse", "outside_rmse"]].agg(["mean", "std"]).round(4)
summ.to_csv(OUT_T / "downstream_ref_summary.csv")
pd.DataFrame(base_rows).round(4).to_csv(OUT_T / "downstream_baselines.csv", index=False)
pd.DataFrame(infer_rows).round(4).to_csv(OUT_T / "downstream_temporal_inference.csv", index=False)
pd.DataFrame(decomp_rows).round(4).to_csv(OUT_T / "downstream_support_decomposition.csv", index=False)
print(summ); print(pd.DataFrame(infer_rows).round(3))

# ---------------------------------------------------------------- ablation of > 12 h treatment (H = 1)
H = 1; tidx = test_index(H); yte, base, _ = baselines(tidx, H); Xte = np.stack([v_orig[i - SEQ:i] for i in tidx])
rmse_p = metrics(base["persistence"], yte)["rmse"]
inside = (yte >= SUPPORT[0]) & (yte <= SUPPORT[1])
abl_rows = []
in_long_arr = in_long.to_numpy()
for name, m in [("rules_repair", mtr), ("ae_repair", ae_mask)]:
    cfgs = {
        "(ii) diurnal fallback (default)": (pd.DataFrame({s: repair(m[s], s) for s in FEATURES}), None),
        "(i) exclude windows intersecting >12 h outages": (pd.DataFrame({s: repair(m[s], s) for s in FEATURES}),
                                                          lambda i: not in_long_arr[i - SEQ:i + H].any()),
        "(iii) no long-gap fill, drop affected windows": (pd.DataFrame({s: repair(m[s], s, long_fill="none") for s in FEATURES}), None),
    }
    for cfg, (df, wf) in cfgs.items():
        preds, idx = train_eval(df, name, H, tidx, Xte, yte, window_filter=wf)
        for seed, p in zip(SEEDS, preds):
            mm = metrics(p, yte); abl_rows.append(dict(variant=name, config=cfg, seed=seed, n_train=len(idx), **mm, skill=1 - mm["rmse"] / rmse_p,
                                                       inside_rmse=float(np.sqrt(np.mean((p - yte)[inside] ** 2))), outside_rmse=float(np.sqrt(np.mean((p - yte)[~inside] ** 2)))))
        print(name, cfg, len(idx), np.mean([r["rmse"] for r in abl_rows if r["variant"] == name and r["config"] == cfg]))
abl = pd.DataFrame(abl_rows); abl.to_csv(OUT_T / "downstream_longgap_ablation_raw.csv", index=False)
abl.groupby(["variant", "config"])[["rmse", "mae", "r2", "skill", "inside_rmse", "outside_rmse", "n_train"]].agg(["mean", "std"]).round(4).to_csv(OUT_T / "downstream_longgap_ablation.csv")

# ---------------------------------------------------------------- figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
order = VARIANTS; labels = ["raw", "rules\n$\\rightarrow$repair", "LSTM-AE\n$\\rightarrow$repair", "union\n$\\rightarrow$repair", "drop-\nfaulty"]
bdf = pd.DataFrame(base_rows)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=False)
for ax, H in zip(axes, HORIZONS):
    m = res[res.H == H].groupby("variant").rmse.agg(["mean", "std"]).reindex(order)
    ax.bar(range(5), m["mean"], yerr=m["std"], capsize=3, color=["tab:red", "tab:green", "tab:olive", "tab:blue", "tab:gray"])
    pr = bdf[(bdf.H == H) & (bdf.variant == "persistence")].rmse.iloc[0]
    ax.axhline(pr, color="black", ls="--", lw=1, label=f"persistence ({pr:.2f})")
    ax.set_xticks(range(5), labels, fontsize=7); ax.set_title(f"horizon {H*5} min" if H < 12 else f"horizon {H*5//60} h", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
axes[0].set_ylabel("RMSE (index units)")
fig.suptitle("Forecast RMSE by training-data variant vs. the persistence reference (10 seeds, mean $\\pm$ sd)", y=1.02)
fig.tight_layout(); fig.savefig(OUT_F / "fig10_downstream_variants.png"); fig.savefig(OUT_F / "fig10_downstream_variants.pdf"); plt.close(fig)

# decomposition figure (H = 1)
dd = pd.DataFrame(decomp_rows); d1 = dd[dd.H == 1].set_index("variant").reindex(["persistence"] + order)
fig, ax = plt.subplots(figsize=(7, 3.2)); x = np.arange(len(d1)); w = 0.38
ax.bar(x - w / 2, d1.inside_rmse, w, label=f"inside training support ({SUPPORT[0]:.0f}-{SUPPORT[1]:.0f} index units)", color="tab:blue")
ax.bar(x + w / 2, d1.outside_rmse, w, label="outside training support", color="tab:orange")
ax.set_xticks(x, ["persistence"] + labels, fontsize=9); ax.set_ylabel("RMSE, 5-min horizon (index units)", fontsize=9); ax.legend(fontsize=9)
ax.set_title("Test error inside vs. outside the soil-moisture index range seen in training", fontsize=10)
fig.tight_layout(); fig.savefig(OUT_F / "fig12_support_decomposition.png"); fig.savefig(OUT_F / "fig12_support_decomposition.pdf"); plt.close(fig)

# prediction figure (H = 1, seed 42) with the line broken at gaps
ts, yt, _ = store[(1, "raw")]
fig, ax = plt.subplots(figsize=(10, 3.2))
def broken(ts, vals):
    t = pd.DatetimeIndex(ts); v = np.asarray(vals, float).copy()
    out_t, out_v = [t[0]], [v[0]]
    for k in range(1, len(t)):
        if (t[k] - t[k - 1]) > pd.Timedelta(minutes=5):
            out_t.append(t[k - 1] + pd.Timedelta(minutes=5)); out_v.append(np.nan)
        out_t.append(t[k]); out_v.append(v[k])
    return pd.DatetimeIndex(out_t), np.array(out_v)
bt, bv = broken(ts, yt); ax.plot(bt, bv, color="black", lw=1.2, label="observed (gaps broken)")
pt, pv = broken(ts, store[(1, "raw")][2]); ax.plot(pt, pv, color="tab:red", lw=0.8, alpha=0.8, label="raw-trained")
pt, pv = broken(ts, store[(1, "ae_repair")][2]); ax.plot(pt, pv, color="tab:olive", lw=0.8, alpha=0.9, label="LSTM-AE $\\rightarrow$ repair")
pt, pv = broken(ts, y_all[tidx - 1]); ax.plot(pt, pv, color="tab:gray", lw=0.8, ls="--", alpha=0.8, label="persistence")
ax.axhspan(SUPPORT[0], SUPPORT[1], color="tab:blue", alpha=0.07, label="training support")
ax.set_ylabel("Soil-moisture index (index units)"); ax.legend(fontsize=8, ncol=5, loc="upper right"); ax.set_title("5-min-ahead predictions on the common untouched test set (seed 42)", fontsize=9)
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT_F / "fig11_downstream_predictions.png"); fig.savefig(OUT_F / "fig11_downstream_predictions.pdf"); plt.close(fig)

headline = {"flags": flag_stats, "long_gap": longgap_stats, "support": SUPPORT, "test_sets": {str(k): v for k, v in test_stats.items()},
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "tensorflow": tf.__version__, "platform": platform.platform(),
                            "op_determinism": True, "block_len": BLOCK, "n_boot": N_BOOT}}
with open(OUT_T / "downstream_ref_headline.json", "w") as f: json.dump(headline, f, indent=2)
print(json.dumps(headline, indent=2)); print("done")
