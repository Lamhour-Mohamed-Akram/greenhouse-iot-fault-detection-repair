# Script 17 - Random-mask control for rules-guided repair (protocol: results/random_mask_control/PROTOCOL.md)
#
# Compares rules->repair with random-mask->repair on the main chronological split at the 5-min horizon.
# The random control masks, on the training partition only, the same number of observed values per channel
# as the rule detector, in runs of the same lengths, at random positions; genuine missing values are masked
# identically in both conditions. Everything else (repair, normalization, model, training, test set) is the
# pipeline of script 10.
#
# Phase 1 (always): build the masks and export the matching diagnostics.
# Phase 2 (unless MASKS_ONLY=1): train 10 seeds for raw and rules->repair (reproduction check against
# script 10) and 5 mask seeds x 10 training seeds for the random control, saving every prediction.
# Phase 3: metrics, paired differences and temporal inference.
import json, os, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from scipy import stats

tf.config.experimental.enable_op_determinism()
ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
OUT = ROOT / "results" / "random_mask_control"; PRED = OUT / "predictions"; OUT.mkdir(parents=True, exist_ok=True); PRED.mkdir(exist_ok=True)
REF_T = ROOT / "results" / "tables"
DATA = ROOT / "data" / "data.csv.gz"
SEEDS = list(range(42, 52)); MASK_SEEDS = [101, 202, 303, 404, 505]
TARGET = "soil_moisture"; FEATURES = ["humidity", "temperature", "soil_moisture"]
FREQ = "5min"; SEQ = 12; H = 1
VALID = {"humidity": (0, 100), "temperature": (-5, 55), "soil_moisture": (0, 100)}
LINEAR_MAX = 48; EPOCHS, BATCH = 30, 256
N_BOOT = 2000; RNG = np.random.default_rng(0)
LOG = open(OUT / "run_log.txt", "a")
def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"; print(line); LOG.write(line + "\n"); LOG.flush()

raw = pd.read_csv(DATA, encoding="latin-1")
raw.columns = ["Id", "humidity", "temperature", "soil_moisture", "soil_temp", "co2", "light", "date"]
raw["date"] = pd.to_datetime(raw["date"]); raw = raw.sort_values("date").reset_index(drop=True)
grid = raw.set_index("date")[FEATURES].resample(FREQ).median()
split = int(0.7 * len(grid)); gtr, gte = grid.iloc[:split], grid.iloc[split:]
log(f"grid {len(grid)} rows, split at {grid.index[split]}, train {split}, test {len(gte)}")

# ---------------------------------------------------------------- rule masks (as script 10)
_STUCK = {}
def raw_stuck_window(s, t0, t1):
    key = (s, t0, t1)
    if key not in _STUCK:
        r = raw[(raw["date"] >= t0) & (raw["date"] < t1)]
        x = r[s]; d = r["date"]; gap = d.diff().dt.total_seconds() > 300
        g = ((x.diff() != 0) | gap).cumsum(); st = d.groupby(g).transform("first"); en = d.groupby(g).transform("last")
        _STUCK[key] = pd.Series(((en - st).dt.total_seconds() > 3600).values, index=d).groupby(pd.Grouper(freq=FREQ)).any()
    return _STUCK[key]
def rule_mask(x, s):
    lo, hi = VALID[s]; oor = ((x < lo) | (x > hi)) & x.notna()
    stuck = raw_stuck_window(s, x.index[0], x.index[-1] + pd.Timedelta(FREQ)).reindex(x.index, fill_value=False) & x.notna()
    med = x.rolling(61, center=True, min_periods=10).median(); mad = (x - med).abs().rolling(61, center=True, min_periods=10).median()
    spike = ((x - med).abs() > 6 * 1.4826 * mad) & mad.gt(0)
    return oor | stuck | spike | x.isna()
mtr = pd.DataFrame({s: rule_mask(gtr[s], s) for s in FEATURES})
mte = pd.DataFrame({s: rule_mask(gte[s], s) for s in FEATURES})
masks = pd.concat([mtr, mte])

# ---------------------------------------------------------------- repair (as script 10)
def clean_tr(s): return gtr[s].where(~mtr[s])
profiles, offsets = {}, {}
for s in FEATURES:
    xc = clean_tr(s)
    prof = xc.groupby(xc.index.hour * 60 + xc.index.minute).mean().reindex(range(0, 1440, 5)).interpolate(limit_direction="both")
    profiles[s] = prof
    pf = pd.Series(prof.reindex(xc.index.hour * 60 + xc.index.minute).values, index=xc.index); offsets[s] = float((xc - pf).mean())
def gap_lengths(mask):
    m = pd.Series(mask, index=gtr.index); gid = m.ne(m.shift()).cumsum()
    return m.groupby(gid).transform("sum").where(m, 0).to_numpy()
def repair(mask, s):
    y = gtr[s].where(~mask)
    gid = y.isna().ne(y.isna().shift()).cumsum(); gl = y.isna().groupby(gid).transform("sum").where(y.isna(), 0)
    lin = y.interpolate("linear", limit_direction="both")
    pf = pd.Series(profiles[s].reindex(y.index.hour * 60 + y.index.minute).values, index=y.index) + offsets[s]
    return lin.where(gl <= LINEAR_MAX, y.fillna(pf))

# ---------------------------------------------------------------- phase 1: random masks and diagnostics
def runs_of(b):
    """(start, length) of the maximal True blocks of a boolean array."""
    b = np.asarray(b, bool); d = np.diff(np.r_[0, b.astype(int), 0]); st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
    return list(zip(st, en - st))
def free_run_from(ok):
    """For each position, the number of consecutive True values starting there."""
    ok = pd.Series(np.asarray(ok, bool)); grp = (~ok).cumsum()
    return np.where(ok, ok.groupby(grp).cumcount(ascending=False) + 1, 0)
def place(lengths, eligible, rng):
    n = len(eligible); occ = np.zeros(n, bool); notes = []; failed = []
    for L in sorted(lengths, reverse=True):
        dil = occ.copy(); dil[1:] |= occ[:-1]; dil[:-1] |= occ[1:]
        starts = np.where(free_run_from(eligible & ~dil) >= L)[0]
        mode = "separated"
        if len(starts) == 0:
            starts = np.where(free_run_from(eligible & ~occ) >= L)[0]; mode = "adjacent allowed"
        if len(starts) == 0:
            failed.append(int(L)); continue
        p = int(rng.choice(starts)); occ[p:p + L] = True
        if mode != "separated": notes.append(int(L))
    return occ, notes, failed

M = {s: gtr[s].isna().to_numpy() for s in FEATURES}
A = {s: (mtr[s].to_numpy() & ~M[s]) for s in FEATURES}
random_masks, random_masks_B, diag_rows, run_rows = {}, {}, [], []
obs_lens = sorted([L for _, L in runs_of(~M[FEATURES[0]])], reverse=True)
MAX_PIECE = int(obs_lens[1])   # second-longest observed training stretch (identical missingness in all channels)
def split_long(lengths):
    out = []
    for L in lengths:
        if L <= MAX_PIECE: out.append(L); continue
        k = int(np.ceil(L / MAX_PIECE)); base = L // k; out += [base + (1 if i < L - base * k else 0) for i in range(k)]
    return out
def summarize(name, s, extra, union):
    rl = [L for _, L in runs_of(extra)]; ul = [L for _, L in runs_of(union)]
    gl = gap_lengths(union)
    return {"condition": name, "channel": s, "additional_count": int(extra.sum()), "additional_runs": len(rl),
            "additional_run_max": int(max(rl)) if rl else 0, "additional_run_median": float(np.median(rl)) if rl else 0,
            "union_count": int(union.sum()), "union_pct_of_train": round(100 * float(union.mean()), 2), "union_runs": len(ul),
            "union_run_max": int(max(ul)) if ul else 0,
            "repaired_linear_pct": round(100 * float(((gl > 0) & (gl <= LINEAR_MAX)).sum() / max(union.sum(), 1)), 2),
            "repaired_fallback_pct": round(100 * float((gl > LINEAR_MAX).sum() / max(union.sum(), 1)), 2)}
for s in FEATURES:
    diag_rows.append(summarize("rules", s, A[s], M[s] | A[s]) | {"mask_seed": "", "overlap_with_rules_pct": 100.0, "placement_adjacent_runs": 0, "placement_failed_runs": 0, "run_lengths_identical": True})
for ms in MASK_SEEDS:
    random_masks[ms] = {}
    for ci, s in enumerate(FEATURES):
        rng = np.random.default_rng([ms, ci])
        lengths = [L for _, L in runs_of(A[s])]
        R, adj, failed = place(lengths, ~M[s], rng)
        random_masks[ms][s] = R
        same = sorted(L for _, L in runs_of(R)) == sorted(lengths)
        d = summarize("random", s, R, M[s] | R) | {"mask_seed": ms, "overlap_with_rules_pct": round(100 * float((R & A[s]).sum() / max(R.sum(), 1)), 2),
                                                   "placement_adjacent_runs": len(adj), "placement_failed_runs": len(failed), "run_lengths_identical": bool(same)}
        diag_rows.append(d)
        for L in sorted(lengths, reverse=True)[:5]:
            run_rows.append({"mask_seed": ms, "channel": s, "requested_run_length": int(L)})
        if failed: log(f"mask {ms} {s}: FAILED to place runs {failed}")
        if adj: log(f"mask {ms} {s}: placed without separation: {adj}")
        # control B: identical rules, runs longer than MAX_PIECE split into equal pieces
        lb = split_long(lengths); rngB = np.random.default_rng([ms, ci])
        RB, adjB, failedB = place(lb, ~M[s], rngB)
        random_masks_B.setdefault(ms, {})[s] = RB
        dB = summarize("random_split", s, RB, M[s] | RB) | {"mask_seed": ms, "overlap_with_rules_pct": round(100 * float((RB & A[s]).sum() / max(RB.sum(), 1)), 2),
                                                          "placement_adjacent_runs": len(adjB), "placement_failed_runs": len(failedB),
                                                          "run_lengths_identical": bool(sorted(L for _, L in runs_of(RB)) == sorted(lb))}
        diag_rows.append(dB)
        if failedB: log(f"mask B {ms} {s}: FAILED to place runs {failedB}")
        if adjB: log(f"mask B {ms} {s}: placed without separation: {adjB}")
diag = pd.DataFrame(diag_rows)
diag["requested_additional_count"] = diag.channel.map({s: int(A[s].sum()) for s in FEATURES})
diag.to_csv(OUT / "mask_diagnostics.csv", index=False)
np.savez_compressed(OUT / "masks.npz", train_index=gtr.index.astype("int64").to_numpy(),
                    **{f"M_{s}": M[s] for s in FEATURES}, **{f"A_{s}": A[s] for s in FEATURES},
                    **{f"R_{ms}_{s}": random_masks[ms][s] for ms in MASK_SEEDS for s in FEATURES},
                    **{f"RB_{ms}_{s}": random_masks_B[ms][s] for ms in MASK_SEEDS for s in FEATURES})
log("mask diagnostics written"); print(diag.to_string())
config = {"control_B_max_piece_slots": MAX_PIECE, "mask_seeds": MASK_SEEDS, "training_seeds": SEEDS, "horizon_steps": H, "split_time": str(grid.index[split]),
          "linear_max_slots": LINEAR_MAX, "epochs": EPOCHS, "batch": BATCH, "seq_len": SEQ,
          "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                          "tensorflow": tf.__version__, "platform": platform.platform(), "op_determinism": True}}
json.dump(config, open(OUT / "config.json", "w"), indent=2)
if os.environ.get("MASKS_ONLY"): sys.exit(0)

# ---------------------------------------------------------------- phase 2: training
ok_row = ~masks.any(axis=1).values
v_orig = grid.values.astype("float32"); tcol = grid.columns.get_loc(TARGET); y_all = grid[TARGET].to_numpy(dtype="float32")
train_clean_targets = gtr[TARGET][~mtr[TARGET]].dropna(); SUPPORT = (float(train_clean_targets.min()), float(train_clean_targets.max()))
ends = np.arange(SEQ, len(grid) - H + 1)
ok = np.array([ok_row[i - SEQ:i].all() and ok_row[i + H - 1] for i in ends]); tidx = ends[(ends >= split) & ok]
yte = y_all[tidx + H - 1]; pers = y_all[tidx - 1]; ts = grid.index[tidx + H - 1]
Xte = np.stack([v_orig[i - SEQ:i] for i in tidx]); inside = (yte >= SUPPORT[0]) & (yte <= SUPPORT[1])
assert len(tidx) == 1162 and np.isfinite(Xte).all()
np.savez_compressed(OUT / "test_set.npz", ts=ts.astype("int64").to_numpy(), yte=yte, inside=inside, persistence=pers)

def build(n):
    inp = tf.keras.Input((SEQ, n)); out = tf.keras.layers.Dense(1)(tf.keras.layers.LSTM(32)(inp))
    m = tf.keras.Model(inp, out); m.compile("adam", "mse"); return m
def prepare(df):
    v = df.values.astype("float32"); idx = np.arange(SEQ, split - H + 1)
    Xtr = np.stack([v[i - SEQ:i] for i in idx]); ytr = v[idx + H - 1, tcol]
    assert np.isfinite(Xtr).all() and np.isfinite(ytr).all()
    mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-8; ymu, ysd = ytr.mean(), ytr.std()
    return (Xtr - mu) / sd, (ytr - ymu) / ysd, (Xte - mu) / sd, ymu, ysd
def fit_predict(tag, df, seed):
    f = PRED / f"{tag}_seed{seed}.npz"
    if f.exists(): return np.load(f)["pred"]
    try:
        Xs, ys, Xts, ymu, ysd = prepare(df)
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed); m = build(len(FEATURES))
        m.fit(Xs, ys, epochs=EPOCHS, batch_size=BATCH, verbose=0, validation_split=0.1, shuffle=True,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
        p = m.predict(Xts, verbose=0).ravel() * ysd + ymu
    except Exception as e:
        log(f"FAILED fit {tag} seed {seed}: {e!r}"); return None
    np.savez_compressed(f, pred=p, ts=ts.astype("int64").to_numpy(), yte=yte); log(f"fit {tag} seed {seed} saved")
    return p

conditions = {"raw": gtr.ffill().bfill(), "rules_repair": pd.DataFrame({s: repair(mtr[s].to_numpy(), s) for s in FEATURES})}
for ms in MASK_SEEDS:
    conditions[f"random_m{ms}"] = pd.DataFrame({s: repair(M[s] | random_masks[ms][s], s) for s in FEATURES})
for ms in MASK_SEEDS:
    conditions[f"randomsplit_m{ms}"] = pd.DataFrame({s: repair(M[s] | random_masks_B[ms][s], s) for s in FEATURES})
for k, df in conditions.items(): assert np.isfinite(df.values).all(), k
preds, fails = {}, []
for tag, df in conditions.items():
    for seed in SEEDS:
        p = fit_predict(tag, df, seed)
        if p is None: fails.append({"condition": tag, "seed": seed}); continue
        preds[(tag, seed)] = p

# ---------------------------------------------------------------- phase 3: analysis
def m3(p):
    e = p - yte
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "rmse_inside": float(np.sqrt(np.mean(e[inside] ** 2))), "mae_inside": float(np.mean(np.abs(e[inside]))),
            "rmse_outside": float(np.sqrt(np.mean(e[~inside] ** 2))), "mae_outside": float(np.mean(np.abs(e[~inside])))}
rows = []
for (tag, seed), p in preds.items():
    grp = "random_split_repair" if tag.startswith("randomsplit") else ("random_mask_repair" if tag.startswith("random") else tag)
    rows.append({"condition": tag, "group": grp, "mask_seed": int(tag.split("_m")[1]) if tag.startswith("random") else "", "seed": seed, **m3(p)})
rows.append({"condition": "persistence", "group": "persistence", "mask_seed": "", "seed": "", **m3(pers)})
res = pd.DataFrame(rows); res.to_csv(OUT / "per_fit_metrics.csv", index=False)
pd.DataFrame(fails, columns=["condition", "seed"]).to_csv(OUT / "failed_fits.csv", index=False)

# reproduction check against script 10
ref = pd.read_csv(REF_T / "downstream_ref_raw.csv"); ref = ref[ref.H == 1]
repro = []
for v in ["raw", "rules_repair"]:
    for seed in SEEDS:
        a = res[(res.condition == v) & (res.seed == seed)].rmse; b = ref[(ref.variant == v) & (ref.seed == seed)].rmse
        if len(a) and len(b): repro.append({"variant": v, "seed": seed, "rmse_here": float(a.iloc[0]), "rmse_script10": float(b.iloc[0]), "abs_diff": abs(float(a.iloc[0]) - float(b.iloc[0]))})
repro = pd.DataFrame(repro); repro.to_csv(OUT / "reproduction_check.csv", index=False)
log(f"reproduction max abs diff: {repro.abs_diff.max():.2e}")

# summaries (mean across runs)
summ = res.groupby("group")[["rmse", "mae", "rmse_inside", "mae_inside", "rmse_outside", "mae_outside"]].agg(["mean", "std", "count"]).round(4)
summ.to_csv(OUT / "summary_mean_across_runs.csv")
rules = res[res.condition == "rules_repair"].set_index("seed")
paired = []
for control in ["random_mask_repair", "random_split_repair"]:
  rnd = res[res.group == control]
  for metric in ["rmse", "rmse_inside", "rmse_outside", "mae"]:
    by_mask = rnd.groupby("mask_seed")[metric].mean() - rules[metric].mean()
    by_seed = rnd.groupby("seed")[metric].mean() - rules[metric]
    cell = rnd.set_index(["mask_seed", "seed"])[metric] - rnd.set_index(["mask_seed", "seed"]).index.get_level_values("seed").map(rules[metric]).to_numpy()
    paired.append({"control": control, "metric": metric, "mean_delta": round(float(cell.mean()), 4),
                   "mask_level_deltas": ";".join(f"{x:.3f}" for x in by_mask.values), "masks_favoring_rules": int((by_mask > 0).sum()),
                   "seed_level_min": round(float(by_seed.min()), 4), "seed_level_max": round(float(by_seed.max()), 4), "seeds_favoring_rules": int((by_seed > 0).sum()),
                   "fits_favoring_rules": int((cell > 0).sum()), "n_fits": int(len(cell))})
paired = pd.DataFrame(paired); paired.to_csv(OUT / "paired_differences.csv", index=False); print(paired.to_string())

# temporal inference on pooled squared losses
def mbb(la, lb, block):
    n = len(la); nb = int(np.ceil(n / block)); out = np.empty(N_BOOT)
    for k in range(N_BOOT):
        st = RNG.integers(0, n - block + 1, nb); sel = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        out[k] = np.sqrt(la[sel].mean()) - np.sqrt(lb[sel].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
def runboot(la, lb, t):
    t = pd.DatetimeIndex(t); br = np.r_[0, np.where(np.diff(t.values).astype("timedelta64[m]").astype(int) > 5)[0] + 1, len(t)]
    segs = [np.arange(br[i], br[i + 1]) for i in range(len(br) - 1)]; out = np.empty(N_BOOT)
    for k in range(N_BOOT):
        sel = np.concatenate([segs[j] for j in RNG.integers(0, len(segs), len(segs))]); out[k] = np.sqrt(la[sel].mean()) - np.sqrt(lb[sel].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(segs)
def dm(la, lb, h=H):
    d = la - lb; n = len(d); d0 = d - d.mean(); L = h + SEQ - 1; var = np.mean(d0 * d0)
    for k in range(1, L + 1): var += 2 * (1 - k / (L + 1)) * np.mean(d0[k:] * d0[:-k])
    dmv = d.mean() / np.sqrt(max(var, 1e-12) / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dmv), float(2 * (1 - stats.t.cdf(abs(dmv), df=n - 1)))
def pooled(tags):
    P = np.array([preds[k] for k in preds if k[0] in tags]); return ((P - yte) ** 2).mean(axis=0), len(P)
l_rules, n_rules = pooled(["rules_repair"]); l_rand, n_rand = pooled([f"random_m{m}" for m in MASK_SEEDS]); l_raw, _ = pooled(["raw"]); l_pers = (pers - yte) ** 2
l_randB, n_randB = pooled([f"randomsplit_m{m}" for m in MASK_SEEDS])
comparisons = [("random_all", l_rand, "rules_repair", l_rules)] + [(f"random_m{m}", pooled([f"random_m{m}"])[0], "rules_repair", l_rules) for m in MASK_SEEDS] + \
              [("random_all", l_rand, "raw", l_raw), ("random_all", l_rand, "persistence", l_pers), ("rules_repair", l_rules, "persistence", l_pers)] + \
              [("random_split_all", l_randB, "rules_repair", l_rules)] + [(f"randomsplit_m{m}", pooled([f"randomsplit_m{m}"])[0], "rules_repair", l_rules) for m in MASK_SEEDS] + \
              [("random_split_all", l_randB, "raw", l_raw)]
tin = []
for scope, sel in [("full", np.ones(len(yte), bool)), ("inside_support", inside), ("outside_support", ~inside)]:
    for a, la, b, lb in comparisons:
        A_, B_ = la[sel], lb[sel]
        row = {"scope": scope, "a": a, "b": b, "n": int(sel.sum()), "pooled_rmse_a": round(float(np.sqrt(A_.mean())), 4),
               "pooled_rmse_b": round(float(np.sqrt(B_.mean())), 4), "delta": round(float(np.sqrt(A_.mean()) - np.sqrt(B_.mean())), 4)}
        for blk in [12, 24, 48]: lo, hi = mbb(A_, B_, blk); row[f"b{blk}_lo"] = round(lo, 4); row[f"b{blk}_hi"] = round(hi, 4)
        lo, hi, ns = runboot(A_, B_, ts[sel]); row["run_lo"] = round(lo, 4); row["run_hi"] = round(hi, 4); row["n_runs"] = ns
        row["dm"], row["dm_p"] = [round(x, 4) for x in dm(A_, B_)]
        tin.append(row)
tin = pd.DataFrame(tin); tin.to_csv(OUT / "temporal_inference.csv", index=False); print(tin.to_string())
json.dump({"n_fits_rules": n_rules, "n_fits_random": n_rand, "n_fits_random_split": n_randB, "n_failed": len(fails), "support": SUPPORT, "n_test": int(len(yte)),
           "n_inside": int(inside.sum()), "reproduction_max_abs_diff": float(repro.abs_diff.max()) if len(repro) else None},
          open(OUT / "headline.json", "w"), indent=2)
log("done")
