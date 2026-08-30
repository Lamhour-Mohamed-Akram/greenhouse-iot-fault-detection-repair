"""Manuscript Figure 12: test predictions for the raw and best (LSTM-AE -> repair)
training-data variants, seed 42, evaluated on the common untouched test set.
Re-executes the deterministic preprocessing of script 05b, then trains the two
models needed for the figure."""
exec(open('scripts/05b_downstream_variants.py').read().split("# %% [markdown]\n# ## 3.")[0])

import numpy as np, tensorflow as tf, matplotlib.pyplot as plt

row_faulty = rules_masks.any(axis=1)
ok_row = ~row_faulty.values
win_ok = np.array([ok_row[i - SEQ_LEN:i + 1].all() for i in range(SEQ_LEN, len(grid))])
end_idx = np.arange(SEQ_LEN, len(grid))
test_idx = end_idx[(end_idx >= split) & win_ok]
v_orig = grid.values.astype("float32")
X_test_common = np.stack([v_orig[i - SEQ_LEN:i] for i in test_idx])
y_test_common = v_orig[test_idx, grid.columns.get_loc(TARGET)]
assert len(y_test_common) == 1012 and np.isfinite(X_test_common).all()

def make_train_xy(df):
    v = df.values.astype("float32")
    idx = np.arange(SEQ_LEN, len(v))
    X = np.stack([v[i - SEQ_LEN:i] for i in idx])
    y = v[idx, df.columns.get_loc(TARGET)]
    return X, y

preds = {}
for name in ["raw", "ae_repair"]:
    Xtr, ytr = make_train_xy(train_variants[name])
    mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-8
    ymu, ysd = ytr.mean(), ytr.std()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    inp = tf.keras.Input((SEQ_LEN, len(FEATURES)))
    m = tf.keras.Model(inp, tf.keras.layers.Dense(1)(tf.keras.layers.LSTM(32)(inp)))
    m.compile("adam", "mse")
    m.fit((Xtr - mu) / sd, (ytr - ymu) / ysd, epochs=EPOCHS, batch_size=BATCH,
          verbose=0, validation_split=0.1, shuffle=True,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    preds[name] = m.predict((X_test_common - mu) / sd, verbose=0).ravel() * ysd + ymu

idx = grid.index[test_idx]
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(idx, y_test_common, lw=1, color="black", label="observed (untouched clean test)")
ax.plot(idx, preds["raw"], lw=0.8, alpha=0.85, color="tab:red",
        label="predicted, raw training data")
ax.plot(idx, preds["ae_repair"], lw=0.8, alpha=0.85, color="tab:green",
        label="predicted, LSTM-AE-repaired training data")
ax.legend(fontsize=8); ax.set_ylabel("Soil moisture (%)"); ax.set_xlabel("Date (2022)")
ax.set_title("Effect of training-data repair on common-test predictions (seed 42, real data)")
fig.savefig(FIG / "fig11_downstream_predictions.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig11_downstream_predictions.pdf", bbox_inches="tight")
print("fig11 regenerated")
