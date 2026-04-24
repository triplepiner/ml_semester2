"""
iter5.py — Surgical: CatBoost with NATIVE categoricals.

Key hypothesis: our preprocessing integer-encodes categoricals before feeding
them to CatBoost. But CatBoost's secret sauce is its internal "ordered target
statistics" encoder. Passing raw strings + `cat_features` typically lifts CAT
by +0.005-0.015 on problems with meaningful high-card categoricals (State,
JobCategory, LoanPurpose etc.). We replace only the CAT OOFs and rerun
stage-2 + ensemble.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

from utils import (ID_COL, OOF_DIR, OUT_DIR, RATE_MAX, RATE_MIN, SEED,
                   TARGET_A, TARGET_B, clip_rate, get_folds, load_data,
                   print_header, set_seed)
from preprocessing import preprocess
from features import engineer_features
from advanced import ridge_blend_classification, ridge_blend_regression

N_CLASSES = 5

# Columns we treat as categorical for CatBoost — raw strings, no encoding.
CAT_FEATURES = ["EducationLevel", "MaritalStatus", "HomeOwnership", "State",
                "EmploymentStatus", "EmployerType", "JobCategory",
                "LoanPurpose", "CollateralType"]


def build_catboost_view(train_fe: pd.DataFrame, test_fe: pd.DataFrame):
    """
    Prepare a CatBoost-friendly view: categoricals stay as strings, numerics
    have NaN → median imputation (+ structural-zero fill). We KEEP all the
    engineered feat_* columns so CAT sees them too.
    """
    cat_cols = [c for c in CAT_FEATURES if c in train_fe.columns]
    drop = [TARGET_A, TARGET_B, ID_COL]
    tr = train_fe.drop(columns=[c for c in drop if c in train_fe.columns]).copy()
    te = test_fe.drop(columns=[c for c in drop if c in test_fe.columns]).copy()

    # Align columns (test might lose targets)
    common = [c for c in tr.columns if c in te.columns]
    tr, te = tr[common], te[common]

    # Impute numerics with train medians. Leave categoricals as strings.
    for c in tr.columns:
        if c in cat_cols:
            tr[c] = tr[c].fillna("NA").astype(str)
            te[c] = te[c].fillna("NA").astype(str)
        else:
            if tr[c].isna().any() or te[c].isna().any():
                fill = 0.0 if c.startswith("feat_") and tr[c].dtype.kind in "iu" else tr[c].median()
                tr[c] = tr[c].fillna(fill)
                te[c] = te[c].fillna(fill)
    # Any leftover non-cat text columns → numeric coerce
    for c in tr.columns:
        if c not in cat_cols and tr[c].dtype == "object":
            tr[c] = pd.to_numeric(tr[c], errors="coerce").fillna(0.0)
            te[c] = pd.to_numeric(te[c], errors="coerce").fillna(0.0)

    cat_idx = [tr.columns.get_loc(c) for c in cat_cols]
    return tr, te, cat_idx, cat_cols


def fit_cat_cls_native(X, y, Xt, folds, cat_idx, iters=4000):
    oof = np.zeros((len(X), N_CLASSES)); tp = np.zeros((len(Xt), N_CLASSES))
    for tri, vai in folds:
        m = CatBoostClassifier(loss_function="MultiClass", classes_count=N_CLASSES,
                               iterations=iters, learning_rate=0.03, depth=8,
                               l2_leaf_reg=5.0, random_strength=0.5,
                               bagging_temperature=0.2, border_count=254,
                               cat_features=cat_idx, random_seed=SEED,
                               verbose=False, allow_writing_files=False,
                               early_stopping_rounds=200)
        m.fit(X.iloc[tri], y.iloc[tri], eval_set=(X.iloc[vai], y.iloc[vai]),
              use_best_model=True)
        oof[vai] = m.predict_proba(X.iloc[vai])
        tp += m.predict_proba(Xt) / len(folds)
    return oof, tp


def fit_cat_reg_native(X, y, Xt, folds, cat_idx, iters=4000):
    oof = np.zeros(len(X)); tp = np.zeros(len(Xt))
    for tri, vai in folds:
        m = CatBoostRegressor(loss_function="RMSE", iterations=iters,
                              learning_rate=0.03, depth=8, l2_leaf_reg=5.0,
                              random_strength=0.5, bagging_temperature=0.2,
                              border_count=254, cat_features=cat_idx,
                              random_seed=SEED, verbose=False,
                              allow_writing_files=False,
                              early_stopping_rounds=200)
        m.fit(X.iloc[tri], y.iloc[tri], eval_set=(X.iloc[vai], y.iloc[vai]),
              use_best_model=True)
        oof[vai] = m.predict(X.iloc[vai])
        tp += m.predict(Xt) / len(folds)
    return oof, tp


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


def main():
    set_seed(SEED)
    t0 = time.time()
    print_header("iter5 — CatBoost with NATIVE categoricals")

    train, test = load_data()
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)

    # CatBoost view: raw strings + numerics
    X_cat_tr, X_cat_te, cat_idx, cat_cols = build_catboost_view(train_fe, test_fe)
    print(f"  CatBoost view: X={X_cat_tr.shape}  cats={cat_cols}")

    # Everyone-else view (numeric only, as before)
    y_tier = train_fe[TARGET_A].astype(int)
    y_rate = train_fe[TARGET_B].astype(float)
    ids_test = test_fe[ID_COL]

    folds = get_folds(y_tier)

    # ---- Train CAT native ----
    print_header("CAT native: Task A")
    t = time.time()
    oof_cat_A, test_cat_A = fit_cat_cls_native(X_cat_tr, y_tier, X_cat_te, folds, cat_idx)
    acc_new = accuracy_score(y_tier, oof_cat_A.argmax(1))
    print(f"  cat_native_A acc={acc_new:.4f}  ({time.time()-t:.0f}s)")
    np.save(OOF_DIR / "cat_native_A_oof.npy", oof_cat_A)
    np.save(OOF_DIR / "cat_native_A_test.npy", test_cat_A)

    print_header("CAT native: Task B")
    t = time.time()
    oof_cat_B, test_cat_B = fit_cat_reg_native(X_cat_tr, y_rate, X_cat_te, folds, cat_idx)
    r2_new = r2_score(y_rate, oof_cat_B)
    print(f"  cat_native_B R²={r2_new:.4f}  ({time.time()-t:.0f}s)")
    np.save(OOF_DIR / "cat_native_B_oof.npy", oof_cat_B)
    np.save(OOF_DIR / "cat_native_B_test.npy", test_cat_B)

    # Compare to iter4 CAT
    old_acc_A = accuracy_score(y_tier, np.load(OOF_DIR / "v4_cat_A_oof.npy").argmax(1))
    old_r2_B = r2_score(y_rate, np.load(OOF_DIR / "v4_cat_B_oof.npy"))
    print(f"  ΔCAT_A = {acc_new - old_acc_A:+.4f}  (iter4 CAT: {old_acc_A:.4f})")
    print(f"  ΔCAT_B = {r2_new - old_r2_B:+.4f}  (iter4 CAT: {old_r2_B:.4f})")

    # ---- Rebuild stage-2 with the new CAT OOFs ----
    print_header("Stage-2 rebuild using native-CAT OOFs")
    # Load iter4 preprocessed training matrix (we need the same numeric features)
    train_raw, test_raw = load_data()
    X_train, X_test, _, _, _ = preprocess(train_fe, test_fe)
    from iter3 import add_multi_te
    X_train, X_test = add_multi_te(X_train, X_test, train_raw, test_raw,
                                   y_tier, y_rate)

    oof_A = {
        "lgb": np.load(OOF_DIR / "v4_lgb_A_oof.npy"),
        "xgb": np.load(OOF_DIR / "v4_xgb_A_oof.npy"),
        "cat": oof_cat_A,   # use the new native CAT
        "lgb_ord": np.load(OOF_DIR / "lgb_ord_A_oof.npy"),
        "two_stage": np.load(OOF_DIR / "two_stage_A_oof.npy"),
    }
    test_A = {
        "lgb": np.load(OOF_DIR / "v4_lgb_A_test.npy"),
        "xgb": np.load(OOF_DIR / "v4_xgb_A_test.npy"),
        "cat": test_cat_A,
        "lgb_ord": np.load(OOF_DIR / "lgb_ord_A_test.npy"),
        "two_stage": np.load(OOF_DIR / "two_stage_A_test.npy"),
    }
    oof_B = {
        "lgb": np.load(OOF_DIR / "v4_lgb_B_oof.npy"),
        "xgb": np.load(OOF_DIR / "v4_xgb_B_oof.npy"),
        "cat": oof_cat_B,
    }
    test_B = {
        "lgb": np.load(OOF_DIR / "v4_lgb_B_test.npy"),
        "xgb": np.load(OOF_DIR / "v4_xgb_B_test.npy"),
        "cat": test_cat_B,
    }

    # Build aug-feature matrix for stage-2
    parts_tr = [X_train.reset_index(drop=True)]
    parts_te = [X_test.reset_index(drop=True)]
    for m, arr in oof_A.items():
        cols = [f"oofA_{m}_p{k}" for k in range(arr.shape[1])]
        parts_tr.append(pd.DataFrame(arr, columns=cols))
        parts_te.append(pd.DataFrame(test_A[m], columns=cols))
    ord_f = np.load(OOF_DIR / "lgb_ord_A_oof_float.npy")
    ord_t = np.load(OOF_DIR / "lgb_ord_A_test_float.npy")
    parts_tr.append(pd.DataFrame({"oofA_ord_float": ord_f}))
    parts_te.append(pd.DataFrame({"oofA_ord_float": ord_t}))
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
    from stack import (optimise_blend_classification, optimise_blend_regression)

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
    print_header("iter5 FINAL OOF")
    print(f"  Accuracy       : {acc_final:.4f}  ({A_method})")
    print(f"  R²             : {r2_final:.4f}  ({B_method})")
    print(f"  Combined score : {combined:.4f}")
    print(f"  vs The Lions 0.88175 → gap = {combined - 0.88175:+.4f}")

    with open(OOF_DIR / "iter5_weights.json", "w") as f:
        json.dump({"A_method": A_method, "B_method": B_method,
                   "A_convex": {k: float(v) for k, v in w_A.items()},
                   "B_convex": {k: float(v) for k, v in w_B.items()},
                   "acc": acc_final, "r2": r2_final, "combined": combined,
                   "cat_native_gain_A": float(acc_new - old_acc_A),
                   "cat_native_gain_B": float(r2_new - old_r2_B),
                   "task_a_scores": {m: float(accuracy_score(y_tier, oof_A[m].argmax(1)))
                                     for m in oof_A},
                   "task_b_scores": {m: float(r2_score(y_rate, oof_B[m]))
                                     for m in oof_B}},
                  f, indent=2)

    # Only overwrite submission if improvement
    if combined > 0.8389:
        tier_pred = final_A_test.argmax(1).astype(int)
        rate_pred = clip_rate(final_B_test)
        sub = pd.DataFrame({ID_COL: ids_test.astype(int).to_numpy(),
                            TARGET_A: tier_pred, TARGET_B: rate_pred})
        sub.to_csv(OUT_DIR / "submission.csv", index=False)
        print(f"  *** iter5 beats iter3c — submission.csv updated ***")
    else:
        print(f"  iter5 did NOT beat iter3c (0.8389) — keeping previous submission")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
