"""
iter9.py — bundle the cheap remaining ideas:

  (A) Fix the adversarial-AUC=1.00 issue. The culprit is our multi-target
      encoder: train rows get K-fold-noisy encodings, test rows get the
      stable full-train mean. That's a distribution gap. We switch to
      10-fold encoding (less noise per train row) AND inject matched
      noise into the test encodings.

  (B) Group aggregation features: mean/std/P(tier=4) of target within
      (State, LoanPurpose), (State, JobCategory), (EmployerType, LoanPurpose).
      K-fold safe on train, full-train aggregates for test.

  (C) Log-rate Task B model. InterestRate has a hard floor at 4.99 and
      a long tier-4 tail. A model trained on log(rate - 4.98) often
      captures the tier-4 tail better. Predictions are inverted via
      exp(·) + 4.98. Added as a NEW ensemble member, not replacing.

Only the fastest-winning model (CAT) is retrained on the new feature set
for Tasks A and B; the already-strong v6_lgb and v6_xgb OOFs are reused.
Total budget: ~25-35 min.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

from utils import (ID_COL, OOF_DIR, OUT_DIR, RATE_MAX, RATE_MIN, SEED,
                   TARGET_A, TARGET_B, clip_rate, get_folds, load_data,
                   print_header, set_seed)
from preprocessing import preprocess
from features import engineer_features
from advanced import ridge_blend_classification, ridge_blend_regression
from iter7 import load_latest_oofs, stage2_lgb

N_CLASSES = 5


# ---------------------------------------------------------------------------
# (A) multi-target encoding v2 — noise-matched, 10-fold
# ---------------------------------------------------------------------------

def multi_target_encode_v2(train_raw, test_raw, cols, y_tier, y_rate,
                           n_splits=10, smoothing=20.0, seed=SEED,
                           test_noise_scale=1.0):
    """
    Train uses K-fold encoding (fold i encoded from folds ≠ i). Test uses
    full-train means, but we ADD gaussian noise sampled from the per-group
    standard deviation to match the train-side variance. This closes the
    distribution gap that the adversarial validation caught.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    enc_tr = pd.DataFrame(index=train_raw.index)
    enc_te = pd.DataFrame(index=test_raw.index)
    gmean = float(y_rate.mean())
    gt4 = float((y_tier == 4).mean())
    gstd = float(y_rate.std())
    is_tier4 = (y_tier == 4).astype(int)
    rng = np.random.default_rng(seed)

    for c in cols:
        # Train encodings (K-fold)
        mean_col = pd.Series(gmean, index=train_raw.index, dtype=float)
        p4_col = pd.Series(gt4, index=train_raw.index, dtype=float)
        std_col = pd.Series(gstd, index=train_raw.index, dtype=float)
        for tri, vai in skf.split(train_raw, y_tier):
            grp = train_raw[c].iloc[tri]
            agg_r = y_rate.iloc[tri].groupby(grp).agg(["sum", "count", "std"])
            sm_mean = (agg_r["sum"] + smoothing * gmean) / (agg_r["count"] + smoothing)
            sm_std = agg_r["std"].fillna(gstd)
            agg_t = is_tier4.iloc[tri].groupby(grp).agg(["sum", "count"])
            sm_p4 = (agg_t["sum"] + smoothing * gt4) / (agg_t["count"] + smoothing)
            mean_col.iloc[vai] = train_raw[c].iloc[vai].map(sm_mean).fillna(gmean).to_numpy()
            p4_col.iloc[vai] = train_raw[c].iloc[vai].map(sm_p4).fillna(gt4).to_numpy()
            std_col.iloc[vai] = train_raw[c].iloc[vai].map(sm_std).fillna(gstd).to_numpy()
        enc_tr[f"{c}_te_rate"] = mean_col
        enc_tr[f"{c}_te_p4"] = p4_col
        enc_tr[f"{c}_te_std"] = std_col

        # Test encodings (full-train means + matched noise)
        agg_r = y_rate.groupby(train_raw[c]).agg(["sum", "count", "std"])
        sm_mean_te = (agg_r["sum"] + smoothing * gmean) / (agg_r["count"] + smoothing)
        sm_std_te = agg_r["std"].fillna(gstd)
        agg_t = is_tier4.groupby(train_raw[c]).agg(["sum", "count"])
        sm_p4_te = (agg_t["sum"] + smoothing * gt4) / (agg_t["count"] + smoothing)

        # Per-group standard error ≈ sm_std / sqrt(count + smoothing).
        # Match this as the noise magnitude on test.
        base_n = agg_r["count"] + smoothing
        se = sm_std_te / np.sqrt(base_n)

        te_mean = test_raw[c].map(sm_mean_te).fillna(gmean).astype(float)
        te_std = test_raw[c].map(sm_std_te).fillna(gstd).astype(float)
        te_p4 = test_raw[c].map(sm_p4_te).fillna(gt4).astype(float)
        te_se = test_raw[c].map(se).fillna(0.0).astype(float)

        noise = rng.normal(0, 1.0, size=len(test_raw)) * te_se.to_numpy() * test_noise_scale
        enc_te[f"{c}_te_rate"] = te_mean + noise
        enc_te[f"{c}_te_p4"] = te_p4  # probability — don't add noise
        enc_te[f"{c}_te_std"] = te_std
    return enc_tr, enc_te


# ---------------------------------------------------------------------------
# (B) group-aggregation features
# ---------------------------------------------------------------------------

def group_aggregate_features(train_raw, test_raw, y_tier, y_rate,
                             n_splits=10, smoothing=20.0, seed=SEED):
    """
    Aggregations over cross-groups. For each pair, compute target stats
    within the group using K-fold on train and full-train on test.
    """
    pairs = [
        ("State", "LoanPurpose"),
        ("State", "JobCategory"),
        ("EmployerType", "LoanPurpose"),
        ("HomeOwnership", "LoanPurpose"),
    ]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    gmean = float(y_rate.mean())
    gt4 = float((y_tier == 4).mean())
    is_tier4 = (y_tier == 4).astype(int)
    enc_tr = pd.DataFrame(index=train_raw.index)
    enc_te = pd.DataFrame(index=test_raw.index)

    for a, b in pairs:
        if a not in train_raw.columns or b not in train_raw.columns:
            continue
        key = a + "__" + b
        tr_key = train_raw[a].astype(str) + "||" + train_raw[b].astype(str)
        te_key = test_raw[a].astype(str) + "||" + test_raw[b].astype(str)

        mean_col = pd.Series(gmean, index=train_raw.index, dtype=float)
        p4_col = pd.Series(gt4, index=train_raw.index, dtype=float)
        for tri, vai in skf.split(train_raw, y_tier):
            grp = tr_key.iloc[tri]
            agg_r = y_rate.iloc[tri].groupby(grp).agg(["sum", "count"])
            sm_mean = (agg_r["sum"] + smoothing * gmean) / (agg_r["count"] + smoothing)
            agg_t = is_tier4.iloc[tri].groupby(grp).agg(["sum", "count"])
            sm_p4 = (agg_t["sum"] + smoothing * gt4) / (agg_t["count"] + smoothing)
            mean_col.iloc[vai] = tr_key.iloc[vai].map(sm_mean).fillna(gmean).to_numpy()
            p4_col.iloc[vai] = tr_key.iloc[vai].map(sm_p4).fillna(gt4).to_numpy()
        enc_tr[f"gagg_{key}_rate"] = mean_col
        enc_tr[f"gagg_{key}_p4"] = p4_col

        agg_r = y_rate.groupby(tr_key).agg(["sum", "count"])
        sm_mean_te = (agg_r["sum"] + smoothing * gmean) / (agg_r["count"] + smoothing)
        agg_t = is_tier4.groupby(tr_key).agg(["sum", "count"])
        sm_p4_te = (agg_t["sum"] + smoothing * gt4) / (agg_t["count"] + smoothing)
        enc_te[f"gagg_{key}_rate"] = te_key.map(sm_mean_te).fillna(gmean).astype(float)
        enc_te[f"gagg_{key}_p4"] = te_key.map(sm_p4_te).fillna(gt4).astype(float)
    return enc_tr, enc_te


# ---------------------------------------------------------------------------
# CAT fits on the new feature set
# ---------------------------------------------------------------------------

def fit_cat_cls(X, y, Xt, folds, iters=3000):
    oof = np.zeros((len(X), N_CLASSES)); tp = np.zeros((len(Xt), N_CLASSES))
    for tri, vai in folds:
        m = CatBoostClassifier(loss_function="MultiClass", classes_count=N_CLASSES,
                               iterations=iters, learning_rate=0.03, depth=8,
                               l2_leaf_reg=5.0, random_strength=0.5,
                               bagging_temperature=0.2, border_count=254,
                               random_seed=SEED, verbose=False,
                               allow_writing_files=False,
                               early_stopping_rounds=200)
        m.fit(X.iloc[tri], y.iloc[tri], eval_set=(X.iloc[vai], y.iloc[vai]),
              use_best_model=True)
        oof[vai] = m.predict_proba(X.iloc[vai])
        tp += m.predict_proba(Xt) / len(folds)
    return oof, tp


def fit_cat_reg(X, y, Xt, folds, iters=3000):
    oof = np.zeros(len(X)); tp = np.zeros(len(Xt))
    for tri, vai in folds:
        m = CatBoostRegressor(loss_function="RMSE", iterations=iters,
                              learning_rate=0.03, depth=8, l2_leaf_reg=5.0,
                              random_strength=0.5, bagging_temperature=0.2,
                              border_count=254, random_seed=SEED,
                              verbose=False, allow_writing_files=False,
                              early_stopping_rounds=200)
        m.fit(X.iloc[tri], y.iloc[tri], eval_set=(X.iloc[vai], y.iloc[vai]),
              use_best_model=True)
        oof[vai] = m.predict(X.iloc[vai])
        tp += m.predict(Xt) / len(folds)
    return oof, tp


# ---------------------------------------------------------------------------
# (C) log-rate Task B model
# ---------------------------------------------------------------------------

def fit_log_rate_lgb(X, y_rate, Xt, folds, rounds=4500):
    """Predict log(rate − 4.98); invert at the end."""
    y_log = np.log(y_rate - 4.98).to_numpy()   # shape (N,)
    p = dict(objective="regression", metric="rmse", learning_rate=0.02,
             num_leaves=127, min_child_samples=15, feature_fraction=0.75,
             bagging_fraction=0.80, bagging_freq=5, lambda_l1=0.1,
             lambda_l2=1.0, verbose=-1, seed=SEED)
    oof_log = np.zeros(len(X))
    tp_log = np.zeros(len(Xt))
    for tri, vai in folds:
        d_tr = lgb.Dataset(X.iloc[tri], label=y_log[tri])
        d_va = lgb.Dataset(X.iloc[vai], label=y_log[vai])
        m = lgb.train(p, d_tr, rounds, valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        oof_log[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration)
        tp_log += m.predict(Xt, num_iteration=m.best_iteration) / len(folds)
    # Invert to rate space
    oof_rate = np.exp(oof_log) + 4.98
    tp_rate = np.exp(tp_log) + 4.98
    return oof_rate, tp_rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    t0 = time.time()
    print_header("iter9 — multi-TE fix + group aggregations + log-rate model")

    train, test = load_data()
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)
    train_raw, test_raw = load_data()
    # Strip targets and Id, keep categorical columns as strings for raw lookups
    drop = [TARGET_A, TARGET_B, ID_COL]
    tr_raw_cats = train_raw.drop(columns=[c for c in drop if c in train_raw.columns]).copy()
    te_raw_cats = test_raw.drop(columns=[c for c in drop if c in test_raw.columns]).copy()
    for c in tr_raw_cats.columns:
        if tr_raw_cats[c].dtype == object:
            tr_raw_cats[c] = tr_raw_cats[c].fillna("NA").astype(str)
            te_raw_cats[c] = te_raw_cats[c].fillna("NA").astype(str)

    X_train, X_test, y_tier, y_rate, ids_test = preprocess(train_fe, test_fe)
    print(f"  base preprocessed: X_train={X_train.shape}")

    # (A) multi-TE v2
    print_header("(A) multi-target encoding v2 (10-fold + noise-matched)")
    te_cols = [c for c in ["State", "JobCategory", "LoanPurpose",
                           "EmployerType", "EmploymentStatus"]
               if c in tr_raw_cats.columns]
    t = time.time()
    enc_tr, enc_te = multi_target_encode_v2(tr_raw_cats[te_cols],
                                            te_raw_cats[te_cols],
                                            te_cols, y_tier, y_rate,
                                            n_splits=10, smoothing=20)
    print(f"  added {enc_tr.shape[1]} multi-TE v2 columns ({time.time()-t:.0f}s)")

    # (B) Group aggregations
    print_header("(B) group aggregation features")
    t = time.time()
    gagg_tr, gagg_te = group_aggregate_features(tr_raw_cats, te_raw_cats,
                                                y_tier, y_rate,
                                                n_splits=10, smoothing=20)
    print(f"  added {gagg_tr.shape[1]} group-agg columns ({time.time()-t:.0f}s)")

    # Remove old multi-TE columns from X_train to avoid double-encoding.
    # Old ones ended with `_te_rate`, `_te_p4`, `_te_std`, `_te`. Keep others.
    old_te_cols = [c for c in X_train.columns
                   if c.endswith("_te_rate") or c.endswith("_te_p4")
                   or c.endswith("_te_std") or c.endswith("_te")]
    if old_te_cols:
        print(f"  dropping {len(old_te_cols)} old multi-TE columns")
        X_train = X_train.drop(columns=old_te_cols)
        X_test = X_test.drop(columns=old_te_cols)

    # Concatenate new features
    X_train_new = pd.concat([X_train.reset_index(drop=True),
                             enc_tr.reset_index(drop=True),
                             gagg_tr.reset_index(drop=True)], axis=1)
    X_test_new = pd.concat([X_test.reset_index(drop=True),
                            enc_te.reset_index(drop=True),
                            gagg_te.reset_index(drop=True)], axis=1)
    print(f"  enriched X_train={X_train_new.shape}  X_test={X_test_new.shape}")

    folds = get_folds(y_tier)

    # ---- Retrain CAT on new features ----
    print_header("Retrain CAT on enriched features")
    t = time.time()
    oof_cat_A, test_cat_A = fit_cat_cls(X_train_new, y_tier, X_test_new, folds)
    print(f"  cat_A acc={accuracy_score(y_tier, oof_cat_A.argmax(1)):.4f}  ({time.time()-t:.0f}s)")
    np.save(OOF_DIR / "v9_cat_A_oof.npy", oof_cat_A)
    np.save(OOF_DIR / "v9_cat_A_test.npy", test_cat_A)

    t = time.time()
    oof_cat_B, test_cat_B = fit_cat_reg(X_train_new, y_rate, X_test_new, folds)
    print(f"  cat_B R²={r2_score(y_rate, oof_cat_B):.4f}  ({time.time()-t:.0f}s)")
    np.save(OOF_DIR / "v9_cat_B_oof.npy", oof_cat_B)
    np.save(OOF_DIR / "v9_cat_B_test.npy", test_cat_B)

    # ---- (C) Log-rate Task B model ----
    print_header("(C) log-rate Task B model")
    t = time.time()
    oof_log_B, test_log_B = fit_log_rate_lgb(X_train_new, y_rate, X_test_new, folds)
    print(f"  log_rate_B R²={r2_score(y_rate, oof_log_B):.4f}  ({time.time()-t:.0f}s)")
    np.save(OOF_DIR / "v9_log_rate_B_oof.npy", oof_log_B)
    np.save(OOF_DIR / "v9_log_rate_B_test.npy", test_log_B)

    # ---- Assemble base OOFs (reuse existing LGB/XGB, swap in new CAT) ----
    print_header("Assemble OOFs for stage-2")
    oof_A, test_A, oof_B, test_B = load_latest_oofs(OOF_DIR, y_tier, y_rate)
    # Swap in v9 CAT
    oof_A["cat"] = oof_cat_A; test_A["cat"] = test_cat_A
    oof_B["cat"] = oof_cat_B; test_B["cat"] = test_cat_B
    # Add log-rate
    oof_B["log_rate"] = oof_log_B; test_B["log_rate"] = test_log_B

    # ---- Stage-2 (multi-seed) ----
    print_header("Stage-2 (multi-seed) on enriched features")
    parts_tr = [X_train_new.reset_index(drop=True)]
    parts_te = [X_test_new.reset_index(drop=True)]
    for m, arr in oof_A.items():
        cols = [f"oofA_{m}_p{k}" for k in range(arr.shape[1])]
        parts_tr.append(pd.DataFrame(arr, columns=cols))
        parts_te.append(pd.DataFrame(test_A[m], columns=cols))
    try:
        ord_f = np.load(OOF_DIR / "lgb_ord_A_oof_float.npy")
        ord_t = np.load(OOF_DIR / "lgb_ord_A_test_float.npy")
        parts_tr.append(pd.DataFrame({"oofA_ord_float": ord_f}))
        parts_te.append(pd.DataFrame({"oofA_ord_float": ord_t}))
    except FileNotFoundError:
        pass
    for m, arr in oof_B.items():
        parts_tr.append(pd.DataFrame({f"oofB_{m}": arr}))
        parts_te.append(pd.DataFrame({f"oofB_{m}": test_B[m]}))
    X_aug_tr = pd.concat(parts_tr, axis=1)
    X_aug_te = pd.concat(parts_te, axis=1)
    print(f"  X_aug_tr={X_aug_tr.shape}")

    t = time.time()
    s2_A_oof, s2_A_test = stage2_lgb(X_aug_tr, y_tier, X_aug_te, folds, "A")
    print(f"  stage2_A acc={accuracy_score(y_tier, s2_A_oof.argmax(1)):.4f}  ({time.time()-t:.0f}s)")
    t = time.time()
    s2_B_oof, s2_B_test = stage2_lgb(X_aug_tr, y_rate, X_aug_te, folds, "B")
    print(f"  stage2_B  R²={r2_score(y_rate, s2_B_oof):.4f}  ({time.time()-t:.0f}s)")

    oof_A["stack2"] = s2_A_oof; test_A["stack2"] = s2_A_test
    oof_B["stack2"] = s2_B_oof; test_B["stack2"] = s2_B_test

    # ---- Ensemble ----
    print_header("Ensemble")
    from stack import optimise_blend_classification, optimise_blend_regression

    w_A = optimise_blend_classification(oof_A, y_tier.to_numpy())
    fA_conv_oof = sum(w_A[n] * oof_A[n] for n in w_A)
    fA_conv_test = sum(w_A[n] * test_A[n] for n in w_A)
    acc_conv = accuracy_score(y_tier, fA_conv_oof.argmax(1))
    _, rA_oof, rA_test = ridge_blend_classification(oof_A, y_tier.to_numpy(), test_A, alpha=1.0)
    acc_ridge = accuracy_score(y_tier, rA_oof.argmax(1))
    print(f"  A convex acc={acc_conv:.4f}   ridge acc={acc_ridge:.4f}")
    print(f"  A weights: { {k: round(float(v),3) for k,v in w_A.items()} }")
    if acc_ridge > acc_conv:
        final_A_oof, final_A_test = rA_oof, rA_test; acc_final, A_method = acc_ridge, "ridge"
    else:
        final_A_oof, final_A_test = fA_conv_oof, fA_conv_test; acc_final, A_method = acc_conv, "convex"

    w_B = optimise_blend_regression(oof_B, y_rate.to_numpy())
    fB_conv_oof = sum(w_B[n] * oof_B[n] for n in w_B)
    fB_conv_test = sum(w_B[n] * test_B[n] for n in w_B)
    r2_conv = r2_score(y_rate, fB_conv_oof)
    _, _, rB_oof, rB_test = ridge_blend_regression(oof_B, y_rate.to_numpy(), test_B, alpha=1.0)
    r2_ridge = r2_score(y_rate, rB_oof)
    print(f"  B convex R²={r2_conv:.4f}   ridge R²={r2_ridge:.4f}")
    print(f"  B weights: { {k: round(float(v),3) for k,v in w_B.items()} }")
    if r2_ridge > r2_conv:
        final_B_oof, final_B_test = rB_oof, rB_test; r2_final, B_method = r2_ridge, "ridge"
    else:
        final_B_oof, final_B_test = fB_conv_oof, fB_conv_test; r2_final, B_method = r2_conv, "convex"

    combined = 0.5 * acc_final + 0.5 * r2_final
    print_header("iter9 FINAL OOF")
    print(f"  Accuracy       : {acc_final:.4f}  ({A_method})")
    print(f"  R²             : {r2_final:.4f}  ({B_method})")
    print(f"  Combined score : {combined:.4f}")
    print(f"  vs The Lions 0.88175 → gap = {combined - 0.88175:+.4f}")
    print(f"  vs iter8 0.8407     → delta = {combined - 0.8407:+.4f}")

    with open(OOF_DIR / "iter9_weights.json", "w") as f:
        json.dump({
            "A_method": A_method, "B_method": B_method,
            "A_convex": {k: float(v) for k, v in w_A.items()},
            "B_convex": {k: float(v) for k, v in w_B.items()},
            "acc": acc_final, "r2": r2_final, "combined": combined,
        }, f, indent=2)

    best_so_far = 0.8407
    if combined > best_so_far:
        tier_pred = final_A_test.argmax(1).astype(int)
        rate_pred = clip_rate(final_B_test)
        sub = pd.DataFrame({ID_COL: ids_test.astype(int).to_numpy(),
                            TARGET_A: tier_pred, TARGET_B: rate_pred})
        sub.to_csv(OUT_DIR / "submission.csv", index=False)
        print(f"  *** iter9 improves over {best_so_far:.4f} — submission updated ***")
    else:
        print(f"  iter9 did not improve over {best_so_far} — keeping previous submission")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
