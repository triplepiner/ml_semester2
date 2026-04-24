"""
iter6.py — Pseudo-labeling. Full-throttle Path B.

Steps:
  1. Load iter5 ensemble test predictions → generate pseudo-labels for test.
  2. Rank test rows by Task-A max class probability. Take top 5000 with
     confidence >= 0.90 — these are rows the ensemble is very sure about.
  3. Build enlarged training = 35 000 real rows + 5 000 pseudo rows.
     Labels for pseudo rows: tier = argmax of ensemble probability, rate
     = clipped ensemble prediction.
  4. Retrain LGB/XGB/CAT base models for both tasks using 5-fold
     StratifiedKFold on the ENLARGED training (real + pseudo). Critical
     rule: compute OOF metric only on real rows — pseudo rows' "OOF" is
     a fit to fake labels and must not enter the metric.
  5. Rebuild stage-2 on the enlarged feature matrix.
  6. Blend (convex + Ridge, pick the winner).
  7. Generate new submission ONLY if combined OOF improves over 0.8389.

Expected lift: +0.005-0.015 based on Kaggle community results for
pseudo-labeling on problems with cleanly calibrated base models.
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

N_CLASSES = 5
PSEUDO_THRESHOLD = 0.90   # only include test rows where max class prob >= this
PSEUDO_MAX_ROWS = 5000


# Folds for enlarged training: stratify on REAL tier labels; pseudo rows
# get their predicted tier as the stratification key.
def make_folds(n_real: int, n_pseudo: int, y_tier_real, y_tier_pseudo):
    y_stratify = np.concatenate([y_tier_real.to_numpy(), y_tier_pseudo])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return list(skf.split(np.arange(n_real + n_pseudo), y_stratify)), n_real


def score_oof_real(y_oof, y_true, n_real, task: str):
    """Evaluate OOF metric using only the first n_real rows (real train)."""
    if task == "A":
        return accuracy_score(y_true, y_oof[:n_real].argmax(1))
    return r2_score(y_true, y_oof[:n_real])


def fit_lgb_cls(X, y, Xt, folds, rounds=4500):
    p = dict(objective="multiclass", num_class=N_CLASSES, metric="multi_logloss",
             learning_rate=0.02, num_leaves=127, min_child_samples=15,
             feature_fraction=0.75, bagging_fraction=0.80, bagging_freq=5,
             lambda_l1=0.1, lambda_l2=1.0, min_split_gain=0.01,
             verbose=-1, seed=SEED)
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


def fit_lgb_reg(X, y, Xt, folds, rounds=4500):
    p = dict(objective="regression", metric="rmse", learning_rate=0.02,
             num_leaves=127, min_child_samples=15, feature_fraction=0.75,
             bagging_fraction=0.80, bagging_freq=5, lambda_l1=0.1,
             lambda_l2=1.0, min_split_gain=0.01, verbose=-1, seed=SEED)
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
    print_header("iter6 — pseudo-labeling full-throttle")

    # ---- Load data ----
    train, test = load_data()
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)
    train_raw, test_raw = load_data()
    X_train, X_test, y_tier, y_rate, ids_test = preprocess(train_fe, test_fe)
    X_train, X_test = add_multi_te(X_train, X_test, train_raw, test_raw,
                                   y_tier, y_rate)
    print(f"  X_train={X_train.shape}  X_test={X_test.shape}")

    # ---- Generate pseudo-labels from iter5 ensemble ----
    print_header("Pseudo-label generation")
    # Compose iter5's final ensemble on test
    test_cat = np.load(OOF_DIR / "cat_native_A_test.npy")
    test_lgb = np.load(OOF_DIR / "v4_lgb_A_test.npy")
    test_xgb = np.load(OOF_DIR / "v4_xgb_A_test.npy")
    test_ord = np.load(OOF_DIR / "lgb_ord_A_test.npy")
    test_2s = np.load(OOF_DIR / "two_stage_A_test.npy")
    test_s2 = np.load(OOF_DIR / "stack2_A_test.npy")
    w = {"lgb": 0.058, "xgb": 0.0, "cat": 0.090, "lgb_ord": 0.011,
         "two_stage": 0.0, "stack2": 0.841}
    probs_A = (w["lgb"] * test_lgb + w["cat"] * test_cat
               + w["lgb_ord"] * test_ord + w["stack2"] * test_s2)
    probs_A = np.clip(probs_A, 1e-12, None)
    probs_A /= probs_A.sum(axis=1, keepdims=True)

    # Rate ensemble weights from iter5
    test_lgb_B = np.load(OOF_DIR / "v4_lgb_B_test.npy")
    test_cat_B = np.load(OOF_DIR / "cat_native_B_test.npy")
    test_s2_B = np.load(OOF_DIR / "stack2_B_test.npy")
    w_B = {"lgb": 0.059, "cat": 0.178, "stack2": 0.763}
    rate_test = (w_B["lgb"] * test_lgb_B + w_B["cat"] * test_cat_B
                 + w_B["stack2"] * test_s2_B)

    conf = probs_A.max(axis=1)
    tier_pseudo_all = probs_A.argmax(axis=1).astype(int)
    rate_pseudo_all = np.clip(rate_test, RATE_MIN, RATE_MAX)

    mask = conf >= PSEUDO_THRESHOLD
    idx_by_conf = np.argsort(-conf)
    # Take min(mask count, MAX_ROWS) of the most confident rows
    top_idx = idx_by_conf[:PSEUDO_MAX_ROWS]
    top_idx = top_idx[conf[top_idx] >= PSEUDO_THRESHOLD]
    print(f"  confidence >= {PSEUDO_THRESHOLD}: {mask.sum()} rows available")
    print(f"  selected top {len(top_idx)} pseudo rows (thr actual={conf[top_idx].min():.3f})")
    print(f"  pseudo tier distribution: {pd.Series(tier_pseudo_all[top_idx]).value_counts().sort_index().to_dict()}")

    # ---- Enlarged training set ----
    n_real = len(X_train)
    n_pseudo = len(top_idx)
    X_enl = pd.concat([X_train.reset_index(drop=True),
                       X_test.iloc[top_idx].reset_index(drop=True)], axis=0, ignore_index=True)
    y_tier_enl = pd.concat([y_tier.reset_index(drop=True),
                            pd.Series(tier_pseudo_all[top_idx])], axis=0, ignore_index=True)
    y_rate_enl = pd.concat([y_rate.reset_index(drop=True),
                            pd.Series(rate_pseudo_all[top_idx])], axis=0, ignore_index=True)
    print(f"  enlarged training: {X_enl.shape}  (real={n_real}, pseudo={n_pseudo})")

    # Stratified 5-fold on enlarged y_tier
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(len(X_enl)), y_tier_enl))

    # ---- Retrain base models ----
    print_header("Retraining base models on enlarged training")
    oof_A, test_A = {}, {}
    oof_B, test_B = {}, {}

    for name, fn in [("lgb", fit_lgb_cls), ("xgb", fit_xgb_cls), ("cat", fit_cat_cls)]:
        t = time.time()
        oof, tp = fn(X_enl, y_tier_enl, X_test, folds)
        acc_real = score_oof_real(oof, y_tier, n_real, "A")
        oof_A[name] = oof; test_A[name] = tp
        np.save(OOF_DIR / f"v6_{name}_A_oof.npy", oof)
        np.save(OOF_DIR / f"v6_{name}_A_test.npy", tp)
        print(f"  v6_{name}_A acc(real)={acc_real:.4f}  ({time.time()-t:.0f}s)")

    for name, fn in [("lgb", fit_lgb_reg), ("xgb", fit_xgb_reg), ("cat", fit_cat_reg)]:
        t = time.time()
        oof, tp = fn(X_enl, y_rate_enl, X_test, folds)
        r2_real = score_oof_real(oof, y_rate, n_real, "B")
        oof_B[name] = oof; test_B[name] = tp
        np.save(OOF_DIR / f"v6_{name}_B_oof.npy", oof)
        np.save(OOF_DIR / f"v6_{name}_B_test.npy", tp)
        print(f"  v6_{name}_B R²(real)={r2_real:.4f}  ({time.time()-t:.0f}s)")

    # Bring in the legacy ordinal / two-stage OOFs, but only for the real rows.
    # We reconstruct extended-length versions by padding with "unknown" predictions
    # for the pseudo slice (equal probabilities). Since stage-2 uses these as
    # features, the pseudo rows will just have flat prior for those columns.
    legacy = {}
    for name in ["lgb_ord", "two_stage"]:
        oof_real = np.load(OOF_DIR / f"{name}_A_oof.npy")
        test_legacy = np.load(OOF_DIR / f"{name}_A_test.npy")
        # For pseudo rows, use the base-model test prediction as a stand-in
        pad = test_legacy[top_idx]
        legacy[name] = (np.concatenate([oof_real, pad], axis=0), test_legacy)
        oof_A[name] = legacy[name][0]
        test_A[name] = legacy[name][1]

    ord_float_real = np.load(OOF_DIR / "lgb_ord_A_oof_float.npy")
    ord_float_test = np.load(OOF_DIR / "lgb_ord_A_test_float.npy")
    ord_float_enl = np.concatenate([ord_float_real, ord_float_test[top_idx]], axis=0)

    # ---- Stage-2 on enlarged feature matrix ----
    print_header("Stage-2 (multi-seed) on enlarged training")
    parts_tr = [X_enl.reset_index(drop=True)]
    parts_te = [X_test.reset_index(drop=True)]
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

    # ---- Ensemble ----
    print_header("Ensemble (metrics on REAL rows only)")
    # Slice OOFs to real rows for weight optimisation
    oof_A_real = {k: v[:n_real] for k, v in oof_A.items()}
    oof_B_real = {k: v[:n_real] for k, v in oof_B.items()}
    from stack import (optimise_blend_classification, optimise_blend_regression)

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
    print_header("iter6 FINAL OOF (real rows only)")
    print(f"  Accuracy       : {acc_final:.4f}  ({A_method})")
    print(f"  R²             : {r2_final:.4f}  ({B_method})")
    print(f"  Combined score : {combined:.4f}")
    print(f"  vs The Lions 0.88175 → gap = {combined - 0.88175:+.4f}")
    print(f"  vs iter3c 0.8389    → delta = {combined - 0.8389:+.4f}")

    with open(OOF_DIR / "iter6_weights.json", "w") as f:
        json.dump({"A_method": A_method, "B_method": B_method,
                   "A_convex": {k: float(v) for k, v in w_A.items()},
                   "B_convex": {k: float(v) for k, v in w_B.items()},
                   "acc": acc_final, "r2": r2_final, "combined": combined,
                   "n_pseudo": int(n_pseudo),
                   "pseudo_confidence_min": float(conf[top_idx].min()),
                   "task_a_scores": {m: float(score_oof_real(oof_A[m], y_tier, n_real, "A"))
                                     for m in oof_A},
                   "task_b_scores": {m: float(score_oof_real(oof_B[m], y_rate, n_real, "B"))
                                     for m in oof_B}},
                  f, indent=2)

    # Only overwrite submission if improvement
    if combined > 0.8389:
        tier_pred = final_A_test.argmax(1).astype(int)
        rate_pred = clip_rate(final_B_test)
        sub = pd.DataFrame({ID_COL: ids_test.astype(int).to_numpy(),
                            TARGET_A: tier_pred, TARGET_B: rate_pred})
        sub.to_csv(OUT_DIR / "submission.csv", index=False)
        print(f"  *** iter6 beats iter3c — submission.csv updated to {combined:.4f} ***")
    else:
        print(f"  iter6 ({combined:.4f}) did NOT beat iter3c (0.8389) — previous submission kept")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
