"""
iter10.py — Second-round pseudo-labeling on iter9's stronger predictions.

Standard Kaggle technique. Each round of pseudo-labeling uses a model whose
predictions are more accurate than the round before, so the pseudo-labels
themselves get cleaner and the gain compounds (but with diminishing returns).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

from utils import (ID_COL, OOF_DIR, OUT_DIR, RATE_MAX, RATE_MIN, SEED,
                   TARGET_A, TARGET_B, clip_rate, load_data, print_header,
                   set_seed)
from preprocessing import preprocess
from features import engineer_features
from advanced import ridge_blend_classification, ridge_blend_regression
from iter3 import add_multi_te
from iter7 import stage2_lgb
from iter9 import (multi_target_encode_v2, group_aggregate_features,
                   fit_cat_cls, fit_cat_reg, fit_log_rate_lgb)

N_CLASSES = 5
PSEUDO_MAX_ROWS = 6500  # one more than iter6 since predictions are now better
PSEUDO_THRESHOLD = 0.85


def fit_lgb_cls(X, y, Xt, folds, rounds=4500):
    p = dict(objective="multiclass", num_class=N_CLASSES, metric="multi_logloss",
             learning_rate=0.02, num_leaves=127, min_child_samples=15,
             feature_fraction=0.75, bagging_fraction=0.80, bagging_freq=5,
             lambda_l1=0.1, lambda_l2=1.0, verbose=-1, seed=SEED)
    oof = np.zeros((len(X), N_CLASSES)); tp = np.zeros((len(Xt), N_CLASSES))
    for tri, vai in folds:
        m = lgb.train(p, lgb.Dataset(X.iloc[tri], y.iloc[tri]), rounds,
                      valid_sets=[lgb.Dataset(X.iloc[vai], y.iloc[vai])],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        oof[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration)
        tp += m.predict(Xt, num_iteration=m.best_iteration) / len(folds)
    return oof, tp


def fit_xgb_cls(X, y, Xt, folds, rounds=4500):
    p = dict(objective="multi:softprob", num_class=N_CLASSES, eval_metric="mlogloss",
             learning_rate=0.02, max_depth=8, subsample=0.80, colsample_bytree=0.70,
             min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0, gamma=0.01,
             tree_method="hist", random_state=SEED, verbosity=0)
    oof = np.zeros((len(X), N_CLASSES)); tp = np.zeros((len(Xt), N_CLASSES))
    dt = xgb.DMatrix(Xt)
    for tri, vai in folds:
        d_tr = xgb.DMatrix(X.iloc[tri], label=y.iloc[tri])
        d_va = xgb.DMatrix(X.iloc[vai], label=y.iloc[vai])
        m = xgb.train(p, d_tr, rounds, [(d_va, "v")], early_stopping_rounds=200,
                      verbose_eval=False)
        oof[vai] = m.predict(d_va, iteration_range=(0, m.best_iteration + 1))
        tp += m.predict(dt, iteration_range=(0, m.best_iteration + 1)) / len(folds)
    return oof, tp


def fit_lgb_reg(X, y, Xt, folds, rounds=4500):
    p = dict(objective="regression", metric="rmse", learning_rate=0.02,
             num_leaves=127, min_child_samples=15, feature_fraction=0.75,
             bagging_fraction=0.80, bagging_freq=5, lambda_l1=0.1,
             lambda_l2=1.0, verbose=-1, seed=SEED)
    oof = np.zeros(len(X)); tp = np.zeros(len(Xt))
    for tri, vai in folds:
        m = lgb.train(p, lgb.Dataset(X.iloc[tri], y.iloc[tri]), rounds,
                      valid_sets=[lgb.Dataset(X.iloc[vai], y.iloc[vai])],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        oof[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration)
        tp += m.predict(Xt, num_iteration=m.best_iteration) / len(folds)
    return oof, tp


def fit_xgb_reg(X, y, Xt, folds, rounds=4500):
    p = dict(objective="reg:squarederror", eval_metric="rmse", learning_rate=0.02,
             max_depth=8, subsample=0.80, colsample_bytree=0.70, min_child_weight=3,
             reg_alpha=0.1, reg_lambda=1.0, gamma=0.01, tree_method="hist",
             random_state=SEED, verbosity=0)
    oof = np.zeros(len(X)); tp = np.zeros(len(Xt))
    dt = xgb.DMatrix(Xt)
    for tri, vai in folds:
        d_tr = xgb.DMatrix(X.iloc[tri], label=y.iloc[tri])
        d_va = xgb.DMatrix(X.iloc[vai], label=y.iloc[vai])
        m = xgb.train(p, d_tr, rounds, [(d_va, "v")], early_stopping_rounds=200,
                      verbose_eval=False)
        oof[vai] = m.predict(d_va, iteration_range=(0, m.best_iteration + 1))
        tp += m.predict(dt, iteration_range=(0, m.best_iteration + 1)) / len(folds)
    return oof, tp


def score_oof_real(y_oof, y_true, n_real, task):
    if task == "A":
        return accuracy_score(y_true, y_oof[:n_real].argmax(1))
    return r2_score(y_true, y_oof[:n_real])


def main():
    set_seed(SEED)
    t0 = time.time()
    print_header("iter10 — 2nd-round pseudo-labeling on iter9 predictions")

    # ---- Load data + rebuild iter9 feature set ----
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
    print(f"  enriched X_train={X_train_new.shape}")

    # ---- Build iter9 ensemble test predictions (loading saved OOFs) ----
    print_header("Derive pseudo-labels from iter9 ensemble")
    test_cat_A_v9 = np.load(OOF_DIR / "v9_cat_A_test.npy")
    test_cat_B_v9 = np.load(OOF_DIR / "v9_cat_B_test.npy")
    test_log_B_v9 = np.load(OOF_DIR / "v9_log_rate_B_test.npy")
    test_lgb_A_v6 = np.load(OOF_DIR / "v6_lgb_A_test.npy")
    test_xgb_A_v6 = np.load(OOF_DIR / "v6_xgb_A_test.npy")
    test_ord_A = np.load(OOF_DIR / "lgb_ord_A_test.npy")
    test_2s_A = np.load(OOF_DIR / "two_stage_A_test.npy")
    # Load the newer stack2 (iter9's)
    try:
        test_s2_A = np.load(OOF_DIR / "stack2_A_test.npy")
    except Exception:
        test_s2_A = test_cat_A_v9  # safe fallback
    # Load iter9 weights to approximate ensemble weights
    with open(OOF_DIR / "iter9_weights.json") as f:
        w9 = json.load(f)
    wa = w9.get("A_convex", {})
    # Compose the Task-A ensemble probs on test (rough — may miss a model)
    probs_A = (wa.get("lgb", 0) * test_lgb_A_v6
               + wa.get("xgb", 0) * test_xgb_A_v6
               + wa.get("cat", 0) * test_cat_A_v9
               + wa.get("lgb_ord", 0) * test_ord_A
               + wa.get("two_stage", 0) * test_2s_A
               + wa.get("stack2", 0) * test_s2_A)
    probs_A = np.clip(probs_A, 1e-9, None)
    probs_A /= probs_A.sum(axis=1, keepdims=True)

    # Task B ensemble test predictions
    test_lgb_B_v6 = np.load(OOF_DIR / "v6_lgb_B_test.npy")
    test_xgb_B_v6 = np.load(OOF_DIR / "v6_xgb_B_test.npy")
    try:
        test_s2_B = np.load(OOF_DIR / "stack2_B_test.npy")
    except Exception:
        test_s2_B = test_cat_B_v9
    wb = w9.get("B_convex", {})
    rate_test = (wb.get("lgb", 0) * test_lgb_B_v6
                 + wb.get("xgb", 0) * test_xgb_B_v6
                 + wb.get("cat", 0) * test_cat_B_v9
                 + wb.get("log_rate", 0) * test_log_B_v9
                 + wb.get("stack2", 0) * test_s2_B)

    conf = probs_A.max(axis=1)
    tier_pseudo_all = probs_A.argmax(axis=1).astype(int)
    rate_pseudo_all = np.clip(rate_test, RATE_MIN, RATE_MAX)
    idx_by_conf = np.argsort(-conf)
    top_idx = idx_by_conf[:PSEUDO_MAX_ROWS]
    top_idx = top_idx[conf[top_idx] >= PSEUDO_THRESHOLD]
    print(f"  selected {len(top_idx)} pseudo rows (min conf={conf[top_idx].min():.3f})")
    print(f"  pseudo tier distribution: {pd.Series(tier_pseudo_all[top_idx]).value_counts().sort_index().to_dict()}")

    # ---- Enlarged training ----
    n_real = len(X_train_new)
    n_pseudo = len(top_idx)
    X_enl = pd.concat([X_train_new.reset_index(drop=True),
                       X_test_new.iloc[top_idx].reset_index(drop=True)], axis=0, ignore_index=True)
    y_tier_enl = pd.concat([y_tier.reset_index(drop=True),
                            pd.Series(tier_pseudo_all[top_idx])], axis=0, ignore_index=True)
    y_rate_enl = pd.concat([y_rate.reset_index(drop=True),
                            pd.Series(rate_pseudo_all[top_idx])], axis=0, ignore_index=True)
    print(f"  enlarged: {X_enl.shape}  (real={n_real}, pseudo={n_pseudo})")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(len(X_enl)), y_tier_enl))

    # ---- Retrain base models ----
    print_header("Retraining base models on enlarged training")
    oof_A, test_A = {}, {}
    oof_B, test_B = {}, {}

    for name, fn in [("lgb", fit_lgb_cls), ("xgb", fit_xgb_cls), ("cat", fit_cat_cls)]:
        t = time.time()
        oof, tp = fn(X_enl, y_tier_enl, X_test_new, folds)
        acc_real = score_oof_real(oof, y_tier, n_real, "A")
        oof_A[name] = oof; test_A[name] = tp
        np.save(OOF_DIR / f"v10_{name}_A_oof.npy", oof)
        np.save(OOF_DIR / f"v10_{name}_A_test.npy", tp)
        print(f"  v10_{name}_A acc(real)={acc_real:.4f}  ({time.time()-t:.0f}s)")

    for name, fn in [("lgb", fit_lgb_reg), ("xgb", fit_xgb_reg), ("cat", fit_cat_reg)]:
        t = time.time()
        oof, tp = fn(X_enl, y_rate_enl, X_test_new, folds)
        r2_real = score_oof_real(oof, y_rate, n_real, "B")
        oof_B[name] = oof; test_B[name] = tp
        np.save(OOF_DIR / f"v10_{name}_B_oof.npy", oof)
        np.save(OOF_DIR / f"v10_{name}_B_test.npy", tp)
        print(f"  v10_{name}_B R²(real)={r2_real:.4f}  ({time.time()-t:.0f}s)")

    # log-rate B
    t = time.time()
    oof_log, tp_log = fit_log_rate_lgb(X_enl, y_rate_enl, X_test_new, folds)
    r2_log = score_oof_real(oof_log, y_rate, n_real, "B")
    oof_B["log_rate"] = oof_log; test_B["log_rate"] = tp_log
    np.save(OOF_DIR / "v10_log_rate_B_oof.npy", oof_log)
    np.save(OOF_DIR / "v10_log_rate_B_test.npy", tp_log)
    print(f"  v10_log_rate_B R²(real)={r2_log:.4f}  ({time.time()-t:.0f}s)")

    # Legacy Task A tricks (ord, two_stage) — pad pseudo slice with test prediction
    for name in ["lgb_ord", "two_stage"]:
        oof_real = np.load(OOF_DIR / f"{name}_A_oof.npy")
        tp_legacy = np.load(OOF_DIR / f"{name}_A_test.npy")
        pad = tp_legacy[top_idx]
        oof_A[name] = np.concatenate([oof_real, pad], axis=0)
        test_A[name] = tp_legacy
    ord_float_real = np.load(OOF_DIR / "lgb_ord_A_oof_float.npy")
    ord_float_test = np.load(OOF_DIR / "lgb_ord_A_test_float.npy")
    ord_float_enl = np.concatenate([ord_float_real, ord_float_test[top_idx]], axis=0)

    # ---- Stage-2 on enlarged feature matrix ----
    print_header("Stage-2 on enlarged")
    parts_tr = [X_enl.reset_index(drop=True)]
    parts_te = [X_test_new.reset_index(drop=True)]
    for m, arr in oof_A.items():
        cols = [f"oofA_{m}_p{k}" for k in range(arr.shape[1])]
        parts_tr.append(pd.DataFrame(arr, columns=cols))
        parts_te.append(pd.DataFrame(test_A[m], columns=cols))
    parts_tr.append(pd.DataFrame({"oofA_ord_float": ord_float_enl}))
    parts_te.append(pd.DataFrame({"oofA_ord_float": ord_float_test}))
    for m, arr in oof_B.items():
        parts_tr.append(pd.DataFrame({f"oofB_{m}": arr}))
        parts_te.append(pd.DataFrame({f"oofB_{m}": test_B[m]}))
    X_aug_tr = pd.concat(parts_tr, axis=1)
    X_aug_te = pd.concat(parts_te, axis=1)
    print(f"  X_aug_tr={X_aug_tr.shape}")

    t = time.time()
    s2_A_oof, s2_A_test = stage2_lgb(X_aug_tr, y_tier_enl, X_aug_te, folds, "A")
    acc_s2 = score_oof_real(s2_A_oof, y_tier, n_real, "A")
    print(f"  stage2_A acc(real)={acc_s2:.4f}  ({time.time()-t:.0f}s)")

    t = time.time()
    s2_B_oof, s2_B_test = stage2_lgb(X_aug_tr, y_rate_enl, X_aug_te, folds, "B")
    r2_s2 = score_oof_real(s2_B_oof, y_rate, n_real, "B")
    print(f"  stage2_B R²(real)={r2_s2:.4f}  ({time.time()-t:.0f}s)")

    oof_A["stack2"] = s2_A_oof; test_A["stack2"] = s2_A_test
    oof_B["stack2"] = s2_B_oof; test_B["stack2"] = s2_B_test

    # ---- Ensemble (metrics on REAL rows only) ----
    print_header("Ensemble (real rows)")
    oof_A_real = {k: v[:n_real] for k, v in oof_A.items()}
    oof_B_real = {k: v[:n_real] for k, v in oof_B.items()}
    from stack import optimise_blend_classification, optimise_blend_regression

    w_A = optimise_blend_classification(oof_A_real, y_tier.to_numpy())
    fA_conv_oof = sum(w_A[n] * oof_A_real[n] for n in w_A)
    fA_conv_test = sum(w_A[n] * test_A[n] for n in w_A)
    acc_conv = accuracy_score(y_tier, fA_conv_oof.argmax(1))
    _, rA_oof, rA_test = ridge_blend_classification(oof_A_real, y_tier.to_numpy(), test_A, alpha=1.0)
    acc_ridge = accuracy_score(y_tier, rA_oof.argmax(1))
    print(f"  A convex acc={acc_conv:.4f}   ridge acc={acc_ridge:.4f}")
    print(f"  A weights: { {k: round(float(v),3) for k,v in w_A.items()} }")
    if acc_ridge > acc_conv:
        final_A_oof, final_A_test = rA_oof, rA_test; acc_final, A_method = acc_ridge, "ridge"
    else:
        final_A_oof, final_A_test = fA_conv_oof, fA_conv_test; acc_final, A_method = acc_conv, "convex"

    w_B = optimise_blend_regression(oof_B_real, y_rate.to_numpy())
    fB_conv_oof = sum(w_B[n] * oof_B_real[n] for n in w_B)
    fB_conv_test = sum(w_B[n] * test_B[n] for n in w_B)
    r2_conv = r2_score(y_rate, fB_conv_oof)
    _, _, rB_oof, rB_test = ridge_blend_regression(oof_B_real, y_rate.to_numpy(), test_B, alpha=1.0)
    r2_ridge = r2_score(y_rate, rB_oof)
    print(f"  B convex R²={r2_conv:.4f}   ridge R²={r2_ridge:.4f}")
    print(f"  B weights: { {k: round(float(v),3) for k,v in w_B.items()} }")
    if r2_ridge > r2_conv:
        final_B_oof, final_B_test = rB_oof, rB_test; r2_final, B_method = r2_ridge, "ridge"
    else:
        final_B_oof, final_B_test = fB_conv_oof, fB_conv_test; r2_final, B_method = r2_conv, "convex"

    combined = 0.5 * acc_final + 0.5 * r2_final
    print_header("iter10 FINAL OOF (real rows)")
    print(f"  Accuracy       : {acc_final:.4f}  ({A_method})")
    print(f"  R²             : {r2_final:.4f}  ({B_method})")
    print(f"  Combined score : {combined:.4f}")
    print(f"  vs The Lions 0.88175 → gap = {combined - 0.88175:+.4f}")
    print(f"  vs iter8 0.8407     → delta = {combined - 0.8407:+.4f}")

    with open(OOF_DIR / "iter10_weights.json", "w") as f:
        json.dump({"A_method": A_method, "B_method": B_method,
                   "acc": acc_final, "r2": r2_final, "combined": combined,
                   "n_pseudo": int(n_pseudo)}, f, indent=2)

    best_so_far = max(0.8407, json.load(open(OOF_DIR / "iter9_weights.json")).get("combined", 0.8407))
    if combined > best_so_far:
        tier_pred = final_A_test.argmax(1).astype(int)
        rate_pred = clip_rate(final_B_test)
        sub = pd.DataFrame({ID_COL: ids_test.astype(int).to_numpy(),
                            TARGET_A: tier_pred, TARGET_B: rate_pred})
        sub.to_csv(OUT_DIR / "submission.csv", index=False)
        print(f"  *** iter10 improves over {best_so_far:.4f} — submission updated ***")
    else:
        print(f"  iter10 ({combined:.4f}) did not beat {best_so_far:.4f}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
