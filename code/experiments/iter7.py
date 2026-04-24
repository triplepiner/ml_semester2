"""
iter7.py — Adversarial validation + feature pruning.

Hypothesis: some of our 183 features may behave differently in train vs test
(distribution shift). Tree models that split on these features generalise
poorly to the test set even though they look fine in CV. We:

  1. Train a binary classifier to distinguish train rows (label 0) from test
     rows (label 1). If the AUC is meaningfully above 0.5, there's shift.
  2. Inspect the feature importances from this adversarial model. Features
     that are very important for the classifier are "shift suspects".
  3. Build a pruned feature set that drops the top-K shift-suspects.
  4. Rebuild stage-2 and the final ensemble on the pruned feature set.
  5. If the resulting combined OOF improves, keep it as the new submission.

Runs on top of iter5/iter6 OOFs (whichever is freshest on disk).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from utils import (ID_COL, OOF_DIR, OUT_DIR, RATE_MAX, RATE_MIN, SEED,
                   TARGET_A, TARGET_B, clip_rate, get_folds, load_data,
                   print_header, set_seed)
from preprocessing import preprocess
from features import engineer_features
from advanced import ridge_blend_classification, ridge_blend_regression
from iter3 import add_multi_te


N_CLASSES = 5
DROP_TOP_N = 20  # drop this many shift-suspect features


def adversarial_auc_and_importance(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Run 5-fold adversarial CV; return mean AUC + aggregated importances."""
    X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof = np.zeros(len(X))
    importances = np.zeros(X.shape[1])
    p = dict(objective="binary", metric="auc", learning_rate=0.05,
             num_leaves=63, min_child_samples=20, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
             verbose=-1, seed=SEED)

    for tri, vai in skf.split(X, y):
        m = lgb.train(p, lgb.Dataset(X.iloc[tri], y[tri]), 1000,
                      valid_sets=[lgb.Dataset(X.iloc[vai], y[vai])],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration)
        importances += m.feature_importance(importance_type="gain")

    auc = roc_auc_score(y, oof)
    imp = pd.Series(importances / 5, index=X.columns).sort_values(ascending=False)
    return auc, imp


def stage2_lgb(X_aug_tr, y, X_aug_te, folds, task, seeds=(42, 1337, 2024)):
    base = dict(learning_rate=0.025, num_leaves=31, min_child_samples=30,
                feature_fraction=0.8, bagging_fraction=0.85, bagging_freq=5,
                lambda_l2=1.0, verbose=-1)
    if task == "A":
        base.update(objective="multiclass", num_class=N_CLASSES,
                    metric="multi_logloss")
        oof_all = np.zeros((len(X_aug_tr), N_CLASSES))
        test_all = np.zeros((len(X_aug_te), N_CLASSES))
    else:
        base.update(objective="regression", metric="rmse")
        oof_all = np.zeros(len(X_aug_tr))
        test_all = np.zeros(len(X_aug_te))
    for seed in seeds:
        p = {**base, "seed": seed}
        oof = np.zeros_like(oof_all); tp = np.zeros_like(test_all)
        for tri, vai in folds:
            m = lgb.train(p, lgb.Dataset(X_aug_tr.iloc[tri], y.iloc[tri]),
                          3000, valid_sets=[lgb.Dataset(X_aug_tr.iloc[vai], y.iloc[vai])],
                          callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
            oof[vai] = m.predict(X_aug_tr.iloc[vai], num_iteration=m.best_iteration)
            tp += m.predict(X_aug_te, num_iteration=m.best_iteration) / len(folds)
        oof_all += oof / len(seeds); test_all += tp / len(seeds)
    return oof_all, test_all


def load_latest_oofs(oof_dir: Path, y_tier, y_rate):
    """
    Pick iter6 OOFs if present (v6_*), otherwise iter4/iter5. Returns two
    dicts (Task A, Task B) keyed by model name.
    """
    has_v6 = (oof_dir / "v6_cat_A_oof.npy").exists()
    prefix = "v6" if has_v6 else "v4"
    if has_v6:
        print(f"  using iter6 OOFs (prefix v6)")
    else:
        print(f"  using iter4 OOFs (prefix v4)")

    # If v6, it was trained on enlarged (40k) data — we slice to real 35k below.
    oof_A = {}
    test_A = {}
    for m in ["lgb", "xgb", "cat"]:
        arr = np.load(oof_dir / f"{prefix}_{m}_A_oof.npy")
        if len(arr) > len(y_tier):  # enlarged version
            arr = arr[: len(y_tier)]
        oof_A[m] = arr
        test_A[m] = np.load(oof_dir / f"{prefix}_{m}_A_test.npy")

    # Always include legacy tricks if present
    for m in ["lgb_ord", "two_stage"]:
        try:
            arr = np.load(oof_dir / f"{m}_A_oof.npy")
            if len(arr) > len(y_tier):
                arr = arr[: len(y_tier)]
            oof_A[m] = arr
            test_A[m] = np.load(oof_dir / f"{m}_A_test.npy")
        except FileNotFoundError:
            pass

    oof_B = {}
    test_B = {}
    for m in ["lgb", "xgb", "cat"]:
        arr = np.load(oof_dir / f"{prefix}_{m}_B_oof.npy")
        if len(arr) > len(y_rate):
            arr = arr[: len(y_rate)]
        oof_B[m] = arr
        test_B[m] = np.load(oof_dir / f"{prefix}_{m}_B_test.npy")

    return oof_A, test_A, oof_B, test_B


def main():
    set_seed(SEED)
    t0 = time.time()
    print_header("iter7 — adversarial validation + feature pruning")

    train, test = load_data()
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)
    X_train, X_test, y_tier, y_rate, ids_test = preprocess(train_fe, test_fe)
    X_train, X_test = add_multi_te(X_train, X_test, load_data()[0],
                                   load_data()[1], y_tier, y_rate)
    print(f"  X_train={X_train.shape}  X_test={X_test.shape}")

    # ---- Step 1: adversarial validation ----
    print_header("Step 1 — adversarial validation")
    t = time.time()
    auc, imp = adversarial_auc_and_importance(X_train, X_test)
    print(f"  adversarial AUC = {auc:.4f}   ({time.time()-t:.0f}s)")
    print(f"  top-20 shift suspects:")
    for name, gain in imp.head(20).items():
        print(f"    {name:45s}  gain={gain:10.0f}")

    # ---- Step 2: build pruned feature set ----
    if auc < 0.52:
        print("  No meaningful distribution shift detected (AUC ~= 0.5). "
              "Skipping feature pruning.")
        dropped = []
    else:
        dropped = imp.head(DROP_TOP_N).index.tolist()
        print(f"  dropping {len(dropped)} features")
    keep = [c for c in X_train.columns if c not in dropped]
    X_train_p = X_train[keep].copy()
    X_test_p = X_test[keep].copy()
    print(f"  pruned X_train={X_train_p.shape}  X_test={X_test_p.shape}")

    folds = get_folds(y_tier)

    # ---- Step 3: load latest base OOFs ----
    print_header("Step 3 — load latest base OOFs")
    oof_A, test_A, oof_B, test_B = load_latest_oofs(OOF_DIR, y_tier, y_rate)
    for m in oof_A:
        acc = accuracy_score(y_tier, oof_A[m].argmax(1))
        print(f"  A/{m:10s} acc={acc:.4f}")
    for m in oof_B:
        r2 = r2_score(y_rate, oof_B[m])
        print(f"  B/{m:10s} R²={r2:.4f}")

    # ---- Step 4: stage-2 on pruned augmented features ----
    print_header("Step 4 — stage-2 on pruned augmented feature set")
    parts_tr = [X_train_p.reset_index(drop=True)]
    parts_te = [X_test_p.reset_index(drop=True)]
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

    s2_A_oof, s2_A_test = stage2_lgb(X_aug_tr, y_tier, X_aug_te, folds, "A")
    acc_s2 = accuracy_score(y_tier, s2_A_oof.argmax(1))
    print(f"  stage2_A acc={acc_s2:.4f}")

    s2_B_oof, s2_B_test = stage2_lgb(X_aug_tr, y_rate, X_aug_te, folds, "B")
    r2_s2 = r2_score(y_rate, s2_B_oof)
    print(f"  stage2_B  R²={r2_s2:.4f}")

    oof_A["stack2"] = s2_A_oof; test_A["stack2"] = s2_A_test
    oof_B["stack2"] = s2_B_oof; test_B["stack2"] = s2_B_test

    # ---- Step 5: ensemble ----
    print_header("Step 5 — ensemble")
    from stack import optimise_blend_classification, optimise_blend_regression

    w_A = optimise_blend_classification(oof_A, y_tier.to_numpy())
    fA_conv_oof = sum(w_A[n] * oof_A[n] for n in w_A)
    fA_conv_test = sum(w_A[n] * test_A[n] for n in w_A)
    acc_conv = accuracy_score(y_tier, fA_conv_oof.argmax(1))
    _, rA_oof, rA_test = ridge_blend_classification(oof_A, y_tier.to_numpy(),
                                                    test_A, alpha=1.0)
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
    _, _, rB_oof, rB_test = ridge_blend_regression(oof_B, y_rate.to_numpy(),
                                                    test_B, alpha=1.0)
    r2_ridge = r2_score(y_rate, rB_oof)
    print(f"  B convex R²={r2_conv:.4f}   ridge R²={r2_ridge:.4f}")
    print(f"  B weights: { {k: round(float(v),3) for k,v in w_B.items()} }")
    if r2_ridge > r2_conv:
        final_B_oof, final_B_test = rB_oof, rB_test; r2_final, B_method = r2_ridge, "ridge"
    else:
        final_B_oof, final_B_test = fB_conv_oof, fB_conv_test; r2_final, B_method = r2_conv, "convex"

    combined = 0.5 * acc_final + 0.5 * r2_final
    print_header("iter7 FINAL OOF")
    print(f"  Accuracy       : {acc_final:.4f}  ({A_method})")
    print(f"  R²             : {r2_final:.4f}  ({B_method})")
    print(f"  Combined score : {combined:.4f}")
    print(f"  Adversarial AUC : {auc:.4f}  (dropped {len(dropped)} features)")
    print(f"  vs The Lions 0.88175 → gap = {combined - 0.88175:+.4f}")
    print(f"  vs iter3c 0.8389    → delta = {combined - 0.8389:+.4f}")

    with open(OOF_DIR / "iter7_weights.json", "w") as f:
        json.dump({
            "A_method": A_method, "B_method": B_method,
            "A_convex": {k: float(v) for k, v in w_A.items()},
            "B_convex": {k: float(v) for k, v in w_B.items()},
            "acc": acc_final, "r2": r2_final, "combined": combined,
            "adversarial_auc": float(auc),
            "dropped_features": dropped,
            "top_20_suspects": {k: float(v) for k, v in imp.head(20).to_dict().items()},
        }, f, indent=2)

    # Submission only if improvement
    best_so_far = 0.8389
    if combined > best_so_far:
        tier_pred = final_A_test.argmax(1).astype(int)
        rate_pred = clip_rate(final_B_test)
        sub = pd.DataFrame({ID_COL: ids_test.astype(int).to_numpy(),
                            TARGET_A: tier_pred, TARGET_B: rate_pred})
        sub.to_csv(OUT_DIR / "submission.csv", index=False)
        print(f"  *** iter7 improves over {best_so_far:.4f} — submission updated ***")
    else:
        print(f"  iter7 did not improve over {best_so_far} — keeping previous submission")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
