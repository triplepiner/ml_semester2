"""
iter12.py — rate-floor two-stage model for Task B.

The insight: 40% of non-tier-4 rows have rate = exactly 4.99 (the legal
floor). Our regression models can't hit 4.99 — they predict 5.3-6.0 for
those rows, bleeding R². A two-stage predictor:

  Stage 1: binary classifier P(rate_is_floor | x)
  Stage 2: regression model trained on NON-floor rows only
  Final : p_floor · 4.99 + (1 − p_floor) · reg_pred

The regression problem on non-floor rows is cleaner (no point-mass), so the
regression arm improves too. Soft blending lets the model express
uncertainty instead of hard snapping.

Runs on the enriched feature set from iter9. Adds a single new Task B model
to the ensemble; rebuilds stage-2 and blends.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

import lightgbm as lgb

from utils import (ID_COL, OOF_DIR, OUT_DIR, RATE_MAX, RATE_MIN, SEED,
                   TARGET_A, TARGET_B, clip_rate, get_folds, load_data,
                   print_header, set_seed)
from preprocessing import preprocess
from features import engineer_features
from advanced import ridge_blend_classification, ridge_blend_regression
from iter7 import load_latest_oofs, stage2_lgb
from iter9 import (multi_target_encode_v2, group_aggregate_features,
                   fit_cat_cls, fit_cat_reg)

N_CLASSES = 5
FLOOR_THRESHOLD = 5.0   # rate <= 5.0 is treated as "at floor"


def fit_floor_two_stage(X, y_rate, Xt, folds, rounds=4500):
    """
    Two-stage Task B:
      - floor_cls: binary LGB for P(rate ≤ 5.00)
      - reg_nonf: LGB regression on rows where rate > 5.00 only
      - blend:    p_floor · 4.99 + (1 − p_floor) · reg_pred

    Returns (oof_rate, test_rate, diag) where diag has floor-classifier AUC.
    """
    is_floor = (y_rate <= FLOOR_THRESHOLD).astype(int)
    print(f"  rows at floor: {is_floor.sum()} / {len(is_floor)} ({is_floor.mean():.1%})")

    oof_floor = np.zeros(len(X))
    oof_reg = np.zeros(len(X))
    test_floor = np.zeros(len(Xt))
    test_reg = np.zeros(len(Xt))

    p_cls = dict(objective="binary", metric="auc", learning_rate=0.03,
                 num_leaves=127, min_child_samples=15, feature_fraction=0.75,
                 bagging_fraction=0.80, bagging_freq=5, lambda_l2=1.0,
                 verbose=-1, seed=SEED)
    p_reg = dict(objective="regression_l1", metric="rmse", learning_rate=0.02,
                 num_leaves=127, min_child_samples=15, feature_fraction=0.75,
                 bagging_fraction=0.80, bagging_freq=5, lambda_l1=0.1,
                 lambda_l2=1.0, verbose=-1, seed=SEED)

    for fi, (tri, vai) in enumerate(folds):
        # Stage 1: floor classifier
        d_tr = lgb.Dataset(X.iloc[tri], is_floor.iloc[tri], free_raw_data=False)
        d_va = lgb.Dataset(X.iloc[vai], is_floor.iloc[vai], free_raw_data=False)
        cls = lgb.train(p_cls, d_tr, rounds, valid_sets=[d_va],
                        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        p_va = cls.predict(X.iloc[vai], num_iteration=cls.best_iteration)
        p_te = cls.predict(Xt, num_iteration=cls.best_iteration)
        oof_floor[vai] = p_va
        test_floor += p_te / len(folds)

        # Stage 2: regressor on NON-floor training rows only
        non_floor_idx = np.array([i for i in tri if not is_floor.iloc[i]])
        r_tr = lgb.Dataset(X.iloc[non_floor_idx],
                           y_rate.iloc[non_floor_idx], free_raw_data=False)
        # Eval on full val fold (some may be floor, some not)
        r_va = lgb.Dataset(X.iloc[vai], y_rate.iloc[vai], free_raw_data=False)
        reg = lgb.train(p_reg, r_tr, rounds, valid_sets=[r_va],
                        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        r_va_pred = reg.predict(X.iloc[vai], num_iteration=reg.best_iteration)
        r_te_pred = reg.predict(Xt, num_iteration=reg.best_iteration)
        oof_reg[vai] = r_va_pred
        test_reg += r_te_pred / len(folds)

    auc = roc_auc_score(is_floor, oof_floor)
    print(f"  floor classifier OOF AUC = {auc:.4f}")

    # Blend soft
    oof_rate = oof_floor * 4.99 + (1 - oof_floor) * oof_reg
    test_rate = test_floor * 4.99 + (1 - test_floor) * test_reg
    # Ensure within legal range
    oof_rate = np.clip(oof_rate, RATE_MIN, RATE_MAX)
    test_rate = np.clip(test_rate, RATE_MIN, RATE_MAX)

    r2_soft = r2_score(y_rate, oof_rate)
    print(f"  two-stage (soft blend) OOF R² = {r2_soft:.4f}")

    # Also report a HARD snap variant — if floor prob > 0.5 use 4.99 else reg.
    hard = np.where(oof_floor > 0.5, 4.99, oof_reg)
    hard = np.clip(hard, RATE_MIN, RATE_MAX)
    r2_hard = r2_score(y_rate, hard)
    print(f"  two-stage (hard snap) OOF R² = {r2_hard:.4f}")

    return oof_rate, test_rate, dict(auc=float(auc), r2_soft=r2_soft, r2_hard=r2_hard)


def build_enriched_features():
    """Rebuild the enriched feature set used by iter9/10/11."""
    train, test = load_data()
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)
    train_raw, test_raw = load_data()
    drop = [TARGET_A, TARGET_B, ID_COL]
    tr_raw_cats = train_raw.drop(columns=[c for c in drop if c in train_raw.columns]).copy()
    te_raw_cats = test_raw.drop(columns=[c for c in drop if c in test_raw.columns]).copy()
    for c in tr_raw_cats.columns:
        if tr_raw_cats[c].dtype == object:
            tr_raw_cats[c] = tr_raw_cats[c].fillna("NA").astype(str)
            te_raw_cats[c] = te_raw_cats[c].fillna("NA").astype(str)

    X_train, X_test, y_tier, y_rate, ids_test = preprocess(train_fe, test_fe)
    te_cols = [c for c in ["State", "JobCategory", "LoanPurpose",
                           "EmployerType", "EmploymentStatus"]
               if c in tr_raw_cats.columns]
    enc_tr, enc_te = multi_target_encode_v2(tr_raw_cats[te_cols], te_raw_cats[te_cols],
                                            te_cols, y_tier, y_rate, n_splits=10, smoothing=20)
    gagg_tr, gagg_te = group_aggregate_features(tr_raw_cats, te_raw_cats,
                                                y_tier, y_rate, n_splits=10, smoothing=20)
    old_te_cols = [c for c in X_train.columns
                   if c.endswith("_te_rate") or c.endswith("_te_p4")
                   or c.endswith("_te_std") or c.endswith("_te")]
    if old_te_cols:
        X_train = X_train.drop(columns=old_te_cols)
        X_test = X_test.drop(columns=old_te_cols)
    X_train_new = pd.concat([X_train.reset_index(drop=True),
                             enc_tr.reset_index(drop=True),
                             gagg_tr.reset_index(drop=True)], axis=1)
    X_test_new = pd.concat([X_test.reset_index(drop=True),
                            enc_te.reset_index(drop=True),
                            gagg_te.reset_index(drop=True)], axis=1)
    return X_train_new, X_test_new, y_tier, y_rate, ids_test


def main():
    set_seed(SEED)
    t0 = time.time()
    print_header("iter12 — rate-floor two-stage model for Task B")

    X_train_new, X_test_new, y_tier, y_rate, ids_test = build_enriched_features()
    print(f"  enriched X_train={X_train_new.shape}")

    folds = get_folds(y_tier)

    # ---- Train rate-floor two-stage ----
    print_header("Rate-floor two-stage model")
    t = time.time()
    oof_floor_B, test_floor_B, diag = fit_floor_two_stage(
        X_train_new, y_rate, X_test_new, folds)
    print(f"  floor model runtime: {time.time()-t:.0f}s")
    np.save(OOF_DIR / "v12_floor_B_oof.npy", oof_floor_B)
    np.save(OOF_DIR / "v12_floor_B_test.npy", test_floor_B)

    # ---- Assemble OOFs for stage-2 ----
    print_header("Assemble latest OOFs + the new floor model")
    oof_A, test_A, oof_B, test_B = load_latest_oofs(OOF_DIR, y_tier, y_rate)
    # Prefer iter9 CAT & log-rate
    if (OOF_DIR / "v9_cat_A_oof.npy").exists():
        oof_A["cat"] = np.load(OOF_DIR / "v9_cat_A_oof.npy")
        test_A["cat"] = np.load(OOF_DIR / "v9_cat_A_test.npy")
    if (OOF_DIR / "v9_cat_B_oof.npy").exists():
        oof_B["cat"] = np.load(OOF_DIR / "v9_cat_B_oof.npy")
        test_B["cat"] = np.load(OOF_DIR / "v9_cat_B_test.npy")
    if (OOF_DIR / "v9_log_rate_B_oof.npy").exists():
        oof_B["log_rate"] = np.load(OOF_DIR / "v9_log_rate_B_oof.npy")
        test_B["log_rate"] = np.load(OOF_DIR / "v9_log_rate_B_test.npy")
    # Add floor two-stage
    oof_B["floor"] = oof_floor_B
    test_B["floor"] = test_floor_B

    # Quick per-model snapshot
    print("  Task B R² snapshot:")
    for m in oof_B:
        print(f"    {m:10s}  R²={r2_score(y_rate, oof_B[m]):.4f}")

    # ---- Stage-2 ----
    print_header("Stage-2 with rate-floor model added")
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
    print_header("iter12 FINAL OOF")
    print(f"  Accuracy       : {acc_final:.4f}  ({A_method})")
    print(f"  R²             : {r2_final:.4f}  ({B_method})")
    print(f"  Combined score : {combined:.4f}")
    print(f"  Floor AUC={diag['auc']:.4f}  soft R²={diag['r2_soft']:.4f}  hard R²={diag['r2_hard']:.4f}")
    print(f"  vs The Lions 0.88175 → gap = {combined - 0.88175:+.4f}")
    print(f"  vs iter8 0.8407     → delta = {combined - 0.8407:+.4f}")

    with open(OOF_DIR / "iter12_weights.json", "w") as f:
        json.dump({"A_method": A_method, "B_method": B_method,
                   "A_convex": {k: float(v) for k, v in w_A.items()},
                   "B_convex": {k: float(v) for k, v in w_B.items()},
                   "acc": acc_final, "r2": r2_final, "combined": combined,
                   "floor_auc": diag["auc"], "floor_r2_soft": diag["r2_soft"]},
                  f, indent=2)

    best_so_far = 0.8407
    if combined > best_so_far:
        tier_pred = final_A_test.argmax(1).astype(int)
        rate_pred = clip_rate(final_B_test)
        sub = pd.DataFrame({ID_COL: ids_test.astype(int).to_numpy(),
                            TARGET_A: tier_pred, TARGET_B: rate_pred})
        sub.to_csv(OUT_DIR / "submission.csv", index=False)
        print(f"  *** iter12 improves over {best_so_far} — submission.csv updated to {combined:.4f} ***")
    else:
        print(f"  iter12 ({combined:.4f}) did not beat {best_so_far:.4f}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
