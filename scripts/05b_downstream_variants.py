# %% [markdown]
# # Script 05b — Downstream Evaluation (RQ4)
#
# Final downstream evaluation using five training-data variants (raw,
# rules->repair, LSTM-AE->repair, union->repair, drop-faulty), 10 fixed seeds
# (42-51), and strictly partition-bounded preprocessing:
# - The chronological 70/30 split is decided first. Every fitted component
#   (normalization, diurnal profiles, autoencoder weights and threshold) and
#   every training transformation (masking, interpolation, filling) uses the
#   training partition only; no training value depends on a post-split
#   observation.
# - The five models differ ONLY in their training data. All are evaluated on
#   one common, untouched test set taken from the original observed 5-min
#   stream: screened-clean windows in the last 30%, never repaired, filled,
#   or reconstructed by any variant.
# - TensorFlow op determinism is enabled and all RNGs are seeded, so repeated
#   runs are expected to be bitwise identical on the same environment.

# %%
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.experimental.enable_op_determinism()

BASE = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
RESULTS = BASE / "results"
FIG, TAB = RESULTS / "figures", RESULTS / "tables"

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

SEEDS = list(range(42, 52))
TARGET = "soil_moisture"
FEATURES = ["humidity", "temperature", "soil_moisture"]
FREQ = "5min"
SEQ_LEN = 12
VALID_RANGE = {"humidity": (0, 100), "temperature": (-5, 55), "soil_moisture": (0, 100)}
STUCK_THRESH_STEPS = 12
LINEAR_MAX_STEPS = 48      # 4 h at 5-min steps: the benchmark-validated linear range
AE_SEQ = 12
EPOCHS, BATCH = 30, 256

raw = pd.read_csv(BASE / "data" / "data.csv.gz", encoding="latin-1")
raw.columns = ["Id", "humidity", "temperature", "soil_moisture", "soil_temp",
               "co2", "light", "date"]
raw["date"] = pd.to_datetime(raw["date"])
grid = raw.sort_values("date").set_index("date")[FEATURES].resample(FREQ).median()
split = int(0.7 * len(grid))
grid_train = grid.iloc[:split]
grid_test = grid.iloc[split:]
print(f"{FREQ} grid: {len(grid):,} rows; train rows {split}, test rows {len(grid) - split}")

# %% [markdown]
# ## 1. Detector flags (partition-bounded)

# %%
def rule_mask(x, sensor):
    lo, hi = VALID_RANGE[sensor]
    oor = ((x < lo) | (x > hi)) & x.notna()
    grp = (x.diff() != 0).cumsum()
    runlen = x.groupby(grp).transform("size")
    stuck = (runlen > STUCK_THRESH_STEPS) & x.notna()
    med = x.rolling(61, center=True, min_periods=10).median()
    mad = (x - med).abs().rolling(61, center=True, min_periods=10).median()
    spike = ((x - med).abs() > 6 * 1.4826 * mad) & mad.gt(0)
    return oor | stuck | spike | x.isna()

# Rule masks are computed separately on each chronological partition so that
# centered rolling windows and constant-run detection never cross the
# train/test boundary. The rule thresholds are fixed a priori (no fitting).
rules_masks_train = pd.DataFrame({s: rule_mask(grid_train[s], s) for s in FEATURES})
rules_masks_test = pd.DataFrame({s: rule_mask(grid_test[s], s) for s in FEATURES})
rules_masks = pd.concat([rules_masks_train, rules_masks_test])

def train_clean_series(s):
    return grid_train[s].where(~rules_masks_train[s])

def per_timestep_error(ae, z, n, seq):
    idx = np.arange(seq, n + 1)
    X = np.stack([z[i - seq:i] for i in idx])[..., None].astype("float32")
    se = (ae.predict(X, verbose=0) - X) ** 2
    err_sum, cnt = np.zeros(n), np.zeros(n)
    for k, end in enumerate(idx - 1):
        err_sum[end - seq + 1:end + 1] += se[k, :, 0]
        cnt[end - seq + 1:end + 1] += 1
    return err_sum / np.maximum(cnt, 1)

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(42)
ae_mask_train, ae_cov_full = {}, {}
for s in FEATURES:
    xc = train_clean_series(s)
    mu, sd = xc.mean(), xc.std()
    ztr = ((xc - mu) / sd).values
    seqs, seq_starts = [], []
    for i in range(AE_SEQ, len(ztr) + 1):
        w = ztr[i - AE_SEQ:i]
        if not np.isnan(w).any():
            seqs.append(w)
            seq_starts.append(i - AE_SEQ)
    Xtr = np.array(seqs, "float32")[..., None]
    inp = tf.keras.Input((AE_SEQ, 1))
    h = tf.keras.layers.LSTM(32)(inp)
    h = tf.keras.layers.RepeatVector(AE_SEQ)(h)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(
        tf.keras.layers.LSTM(32, return_sequences=True)(h))
    ae = tf.keras.Model(inp, out)
    ae.compile("adam", "mse")
    ae.fit(Xtr, Xtr, epochs=20, batch_size=256, verbose=0, shuffle=True)
    # Threshold: reconstruction errors from fully screened-clean training
    # sequences only, aggregated per timestep across the clean windows that
    # cover it. No zero-filled or flagged position contributes.
    se_tr = (ae.predict(Xtr, verbose=0) - Xtr) ** 2
    err_sum, cnt = np.zeros(len(ztr)), np.zeros(len(ztr))
    for k, st in enumerate(seq_starts):
        err_sum[st:st + AE_SEQ] += se_tr[k, :, 0]
        cnt[st:st + AE_SEQ] += 1
    thr = float(np.quantile(err_sum[cnt > 0] / cnt[cnt > 0], 0.995))
    # Training repair mask: AE inference on the TRAINING PARTITION only, so no
    # reconstruction window crosses the train/test boundary. Filling for the
    # inference input stays inside the training partition.
    z_tr_full = (((grid_train[s] - mu) / sd).ffill().bfill()).values
    err_train = per_timestep_error(ae, z_tr_full, len(z_tr_full), AE_SEQ)
    ae_mask_train[s] = pd.Series((err_train > thr) | grid_train[s].isna().values,
                                 index=grid_train.index)
    # Descriptive full-deployment coverage (reporting only; never feeds
    # training preprocessing).
    z_full = (((grid[s] - mu) / sd).ffill().bfill()).values
    err_full = per_timestep_error(ae, z_full, len(z_full), AE_SEQ)
    ae_cov_full[s] = pd.Series((err_full > thr) | grid[s].isna().values,
                               index=grid.index)
    print(f"{s}: AE trained on {len(Xtr)} clean train seqs, thr={thr:.4f}, "
          f"train-mask {100 * ae_mask_train[s].mean():.1f}%, "
          f"full coverage {100 * ae_cov_full[s].mean():.1f}%")

ae_mask_train = pd.DataFrame(ae_mask_train)
ae_cov_full = pd.DataFrame(ae_cov_full)
union_mask_train = rules_masks_train | ae_mask_train
for name, m in [("rules", rules_masks_train), ("lstm_ae", ae_mask_train),
                ("union", union_mask_train)]:
    print(name, "train-partition flagged % per sensor:",
          (100 * m.mean()).round(1).to_dict())

# %% [markdown]
# ## 2. Training repair (training partition only)

# %%
profiles, offsets = {}, {}
for s in FEATURES:
    xc = train_clean_series(s)
    prof = xc.groupby(xc.index.hour * 60 + xc.index.minute).mean()
    prof = prof.reindex(range(0, 1440, 5)).interpolate(limit_direction="both")
    profiles[s] = prof
    pf = pd.Series(prof.reindex(xc.index.hour * 60 + xc.index.minute).values, index=xc.index)
    offsets[s] = float((xc - pf).mean())

def repair_train(mask, s):
    """Repair one training-partition channel: linear for gaps <= 4 h, diurnal
    profile beyond. Operates on grid_train only; interpolation cannot reach
    post-split observations."""
    y = grid_train[s].where(~mask)
    gap_id = y.isna().ne(y.isna().shift()).cumsum()
    gap_len = y.isna().groupby(gap_id).transform("sum").where(y.isna(), 0)
    lin = y.interpolate("linear", limit_direction="both")
    pf = pd.Series(profiles[s].reindex(y.index.hour * 60 + y.index.minute).values,
                   index=y.index) + offsets[s]
    long_fill = y.fillna(pf)
    return lin.where(gap_len <= LINEAR_MAX_STEPS, long_fill)

# Raw naive filling audit: forward fill within the training partition; any
# leading NaN (nothing observed yet to carry forward) is reported and filled
# backward WITHIN the training partition only.
raw_train = grid_train.ffill()
lead_nan = int(raw_train.isna().sum().sum())
print(f"raw variant: leading train NaNs after ffill = {lead_nan} "
      f"(filled backward inside the training partition)")
raw_train = raw_train.bfill()

train_variants = {
    "raw": raw_train,
    "rules_repair": pd.DataFrame({s: repair_train(rules_masks_train[s], s) for s in FEATURES}),
    "ae_repair": pd.DataFrame({s: repair_train(ae_mask_train[s], s) for s in FEATURES}),
    "union_repair": pd.DataFrame({s: repair_train(union_mask_train[s], s) for s in FEATURES}),
}
for name, df in train_variants.items():
    assert np.isfinite(df.values).all(), f"non-finite training values in {name}"

# %% [markdown]
# ## 3. Common untouched test set and the five models

# %%
# The common test set is built ONCE from the ORIGINAL observed grid: last 30%,
# all 12 input rows plus the target rule-clean in all three channels. It is
# never repaired, filled, or reconstructed by any variant.
row_faulty = rules_masks.any(axis=1)
ok_row = ~row_faulty.values
win_ok = np.array([ok_row[i - SEQ_LEN:i + 1].all() for i in range(SEQ_LEN, len(grid))])
end_idx = np.arange(SEQ_LEN, len(grid))
test_idx = end_idx[(end_idx >= split) & win_ok]

v_orig = grid.values.astype("float32")
X_test_common = np.stack([v_orig[i - SEQ_LEN:i] for i in test_idx])
y_test_common = v_orig[test_idx, grid.columns.get_loc(TARGET)]
test_timestamps_common = grid.index[test_idx]

assert len(y_test_common) == 1012, len(y_test_common)
assert np.isfinite(X_test_common).all()
assert np.isfinite(y_test_common).all()
# the common targets are exactly the original observations
assert np.array_equal(y_test_common,
                      grid[TARGET].to_numpy(dtype="float32")[test_idx])

# quantification of the pre-correction defect: how many common-test cells the
# AE/union training masks would have altered had test data been repaired
for name, m in [("rules", rules_masks), ("lstm_ae/union", rules_masks | ae_cov_full)]:
    mm = (m if name == "rules" else ae_cov_full).to_numpy()
    tgt_col = grid.columns.get_loc(TARGET)
    n_tgt = int(mm[test_idx, tgt_col].sum()) if name != "rules" else int(
        rules_masks.to_numpy()[test_idx, tgt_col].sum())
    n_inp = int(sum(mm[i - SEQ_LEN:i].sum() for i in test_idx)) if name != "rules" else int(
        sum(rules_masks.to_numpy()[i - SEQ_LEN:i].sum() for i in test_idx))
    print(f"[audit] {name}: flags inside common test windows: inputs={n_inp}, targets={n_tgt}")

def make_train_xy(df):
    v = df.values.astype("float32")
    idx = np.arange(SEQ_LEN, len(v))
    X = np.stack([v[i - SEQ_LEN:i] for i in idx])
    y = v[idx, df.columns.get_loc(TARGET)]
    return X, y, idx

def build_lstm(n_feat):
    inp = tf.keras.Input((SEQ_LEN, n_feat))
    out = tf.keras.layers.Dense(1)(tf.keras.layers.LSTM(32)(inp))
    m = tf.keras.Model(inp, out)
    m.compile("adam", "mse")
    return m

# drop-faulty: raw training data, windows containing any rule-flagged row removed
flags_train_row = rules_masks_train.any(axis=1).to_numpy()
def train_selector(variant):
    idx = np.arange(SEQ_LEN, split)
    if variant != "drop_faulty":
        return idx
    keep = np.array([not flags_train_row[i - SEQ_LEN:i + 1].any() for i in idx])
    return idx[keep]

results, preds_store = [], {}
VARIANTS_RUN = ["raw", "rules_repair", "ae_repair", "union_repair", "drop_faulty"]
for name in VARIANTS_RUN:
    df = train_variants["raw"] if name == "drop_faulty" else train_variants[name]
    Xall, yall, idx_all = make_train_xy(df)
    sel = train_selector(name)
    pos = np.searchsorted(idx_all, sel)
    Xtr, ytr = Xall[pos], yall[pos]
    assert np.isfinite(Xtr).all() and np.isfinite(ytr).all()
    # normalization from THIS variant's training data only
    mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-8
    ymu, ysd = ytr.mean(), ytr.std()
    Xs, ys = (Xtr - mu) / sd, (ytr - ymu) / ysd
    Xts = (X_test_common - mu) / sd
    # common-test integrity: every variant scores against the same untouched
    # real observations
    assert Xts.shape[0] == 1012
    assert np.shares_memory(y_test_common, y_test_common)
    for seed in SEEDS:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        m = build_lstm(len(FEATURES))
        m.fit(Xs, ys, epochs=EPOCHS, batch_size=BATCH, verbose=0,
              validation_split=0.1, shuffle=True,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,
                         restore_best_weights=True)])
        p = m.predict(Xts, verbose=0).ravel() * ysd + ymu
        truth = y_test_common
        rmse = float(np.sqrt(np.mean((p - truth) ** 2)))
        results.append({"variant": name, "seed": seed, "rmse": rmse,
                        "mae": float(np.mean(np.abs(p - truth))),
                        "r2": float(1 - np.sum((p - truth) ** 2)
                                    / np.sum((truth - truth.mean()) ** 2)),
                        "n_train": int(len(Xtr)), "n_test": int(len(truth))})
        if seed == SEEDS[0]:
            preds_store[name] = (test_timestamps_common, truth, p)
    done = [r for r in results if r["variant"] == name]
    print(name, "rmse:", round(np.mean([r["rmse"] for r in done]), 3),
          "+/-", round(np.std([r["rmse"] for r in done]), 3),
          "n_train:", done[0]["n_train"])

res = pd.DataFrame(results)
res.to_csv(TAB / "downstream_variants_raw.csv", index=False)
agg = res.groupby("variant")[["rmse", "mae", "r2"]].agg(["mean", "std", "median"]).round(4)
agg.to_csv(TAB / "downstream_variants_summary.csv")
print(agg)

# %% [markdown]
# ## 4. Figures and headline numbers

# %%
order = ["raw", "rules_repair", "ae_repair", "union_repair", "drop_faulty"]
labels = ["raw", "rules\n$\\rightarrow$repair", "LSTM-AE\n$\\rightarrow$repair",
          "union\n$\\rightarrow$repair", "drop-\nfaulty"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, metric in zip(axes, ["rmse", "mae"]):
    m = res.groupby("variant")[metric].agg(["mean", "std"]).reindex(order)
    ax.bar(range(5), m["mean"], yerr=m["std"], capsize=3,
           color=["tab:red", "tab:green", "tab:olive", "tab:blue", "tab:gray"])
    ax.set_xticks(range(5), labels, fontsize=7)
    ax.set_ylabel(metric.upper())
fig.suptitle(f"Soil-moisture forecasting on the common untouched test set "
             f"({len(SEEDS)} seeds, mean $\\pm$ sd)")
fig.tight_layout()
fig.savefig(FIG / "fig10_downstream_variants.png")
fig.savefig(FIG / "fig10_downstream_variants.pdf")
plt.close(fig)

m = res.groupby("variant").rmse.mean()
headline = {
    "n_seeds": len(SEEDS),
    "rmse": {k: round(float(v), 3) for k, v in m.items()},
    "rmse_sd": {k: round(float(v), 3) for k, v in res.groupby("variant").rmse.std().items()},
    "mae": {k: round(float(v), 3) for k, v in res.groupby("variant").mae.mean().items()},
    "mae_sd": {k: round(float(v), 3) for k, v in res.groupby("variant").mae.std().items()},
    "r2": {k: round(float(v), 3) for k, v in res.groupby("variant").r2.mean().items()},
    "r2_sd": {k: round(float(v), 3) for k, v in res.groupby("variant").r2.std().items()},
    "improvement_vs_raw_pct": {k: round(float(100 * (m["raw"] - v) / m["raw"]), 1)
                               for k, v in m.items() if k != "raw"},
    "n_train": {k: int(res[res.variant == k].n_train.iloc[0]) for k in m.index},
    "n_test": int(res.n_test.iloc[0]),
    "train_partition_flagged_pct": {
        "rules": {s: round(100 * float(rules_masks_train[s].mean()), 1) for s in FEATURES},
        "lstm_ae": {s: round(100 * float(ae_mask_train[s].mean()), 1) for s in FEATURES},
        "union": {s: round(100 * float(union_mask_train[s].mean()), 1) for s in FEATURES}},
    "descriptive_full_ae_coverage_pct": {
        s: round(100 * float(ae_cov_full[s].mean()), 1) for s in FEATURES},
    "row_any_fault_pct_rules": round(100 * float(row_faulty.mean()), 1),
    "environment": {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "device": "CPU" if not tf.config.list_physical_devices("GPU") else "GPU",
        "op_determinism": True,
        "platform": platform.platform(),
    },
}
with open(TAB / "downstream_variants_headline.json", "w") as f:
    json.dump(headline, f, indent=2)
print(json.dumps(headline, indent=2))
