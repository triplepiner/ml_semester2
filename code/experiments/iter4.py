"""
iter4.py — harder push: strengthen the strong models, drop the weak.

Strategy:
  1. Retrain Task B base models at LR 0.02 with full 6000 rounds + more leaves
     (the undertrained iter3 versions left 0.005-0.010 R² on the table).
  2. Strong Task A: retrain LGB/XGB at LR 0.02 / 6000 rounds.
  3. Multi-seed stage-2: train 3 stage-2 LGBMs with different seeds, average.
  4. Drop broken models (DART, mono variants) from stage-2 features.
  5. Convex + Ridge blend as before, pick winner.
  6. Pseudo-labeling: take the top 3000 high-confidence test rows, add to
     training, retrain stage-2 one more time.

Time budget: ~50-70 min.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

from utils import (ID_COL, OOF_DIR, OUT_DIR, RATE_MAX, RATE_MIN, SEED,
                   TARGET_A, TARGET_B, clip_rate, get_folds, load_data,
                   print_header, set_seed)
from preprocessing import preprocess
from features import engineer_features
from advanced import ridge_blend_classification, ridge_blend_regression
from iter3 import add_multi_te

N_CLASSES = 5


def fit_lgb_cls(X, y, Xt, folds, rounds=6000):
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


def fit_xgb_cls(X, y, Xt, folds, rounds=6000):
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


def fit_cat_cls(X, y, Xt, folds, iters=6000):
    oof = np.zeros((len(X), N_CLASSES)); tp = np.zeros((len(Xt), N_CLASSES))
    for tri, vai in folds:
        m = CatBoostClassifier(loss_function="MultiClass", classes_count=N_CLASSES,
                               iterations=iters, learning_rate=0.03, depth=8,
                               l2_leaf_reg=5.0, random_strength=0.5,
                               bagging_temperature=0.2, border_count=254,
                               random_seed=SEED, verbose=False,
                               allow_writing_files=False, early_stopping_rounds=200)
        m.fit(X.iloc[tri], y.iloc[tri], eval_set=(X.iloc[vai], y.iloc[vai]),
              use_best_model=True)
        oof[vai] = m.predict_proba(X.iloc[vai])
        tp += m.predict_proba(Xt) / len(folds)
    return oof, tp


def fit_lgb_reg(X, y, Xt, folds, rounds=6000):
    p = dict(objective="regression", metric="rmse",
             learning_rate=0.02, num_leaves=127, min_child_samples=15,
             feature_fraction=0.75, bagging_fraction=0.80, bagging_freq=5,
             lambda_l1=0.1, lambda_l2=1.0, min_split_gain=0.01,
             verbose=-1, seed=SEED)
    oof = np.zeros(len(X)); tp = np.zeros(len(Xt))
    for tri, vai in folds:
        m = lgb.train(p, lgb.Dataset(X.iloc[tri], y.iloc[tri]), rounds,
                      valid_sets=[lgb.Dataset(X.iloc[vai], y.iloc[vai])],
                      callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        oof[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration)
        tp += m.predict(Xt, num_iteration=m.best_iteration) / len(folds)
    return oof, tp


def fit_xgb_reg(X, y, Xt, folds, rounds=6000):
    p = dict(objective="reg:squarederror", eval_metric="rmse",
             learning_rate=0.02, max_depth=8, subsample=0.80, colsample_bytree=0.70,
             min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0, gamma=0.01,
             tree_method="hist", random_state=SEED, verbosity=0)
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


def fit_cat_reg(X, y, Xt, folds, iters=6000):
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


def stage2_multi_seed(X_aug_tr, y, X_aug_te, folds, task: str,
                      seeds=(42, 1337, 2024)):
    """Train one stage-2 LGBM per seed; return averaged OOF + test."""
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
    print_header("iter4 — strengthen strong, drop weak")

    train_fe = engineer_features(load_data()[0])
    test_fe = engineer_features(load_data()[1])
    train_raw, test_raw = load_data()
    X_train, X_test, y_tier, y_rate, ids_test = preprocess(train_fe, test_fe)
    X_train, X_test = add_multi_te(X_train, X_test, train_raw, test_raw,
                                   y_tier, y_rate)
    print(f"  X_train={X_train.shape}")

    folds = get_folds(y_tier)

    # ==================== TASK A ====================
    print_header("Task A — retrain LGB/XGB/CAT at LR 0.02 with 6000 rounds")
    oof_A, test_A = {}, {}

    for name, fn in [("lgb", fit_lgb_cls), ("xgb", fit_xgb_cls), ("cat", fit_cat_cls)]:
        t = time.time()
        oof_A[name], test_A[name] = fn(X_train, y_tier, X_test, folds)
        acc = accuracy_score(y_tier, oof_A[name].argmax(1))
        np.save(OOF_DIR / f"v4_{name}_A_oof.npy", oof_A[name])
        np.save(OOF_DIR / f"v4_{name}_A_test.npy", test_A[name])
        print(f"  v4_{name}_A acc={acc:.4f}  ({time.time()-t:.0f}s)")

    # Keep the ordinal regressor trick (load from disk — still valid)
    oof_A["lgb_ord"] = np.load(OOF_DIR / "lgb_ord_A_oof.npy")
    test_A["lgb_ord"] = np.load(OOF_DIR / "lgb_ord_A_test.npy")
    oof_A["two_stage"] = np.load(OOF_DIR / "two_stage_A_oof.npy")
    test_A["two_stage"] = np.load(OOF_DIR / "two_stage_A_test.npy")
    print(f"  [loaded] lgb_ord acc={accuracy_score(y_tier, oof_A['lgb_ord'].argmax(1)):.4f}")
    print(f"  [loaded] two_stage acc={accuracy_score(y_tier, oof_A['two_stage'].argmax(1)):.4f}")

    # ==================== TASK B ====================
    print_header("Task B — retrain all at LR 0.02 / 6000 rounds")
    oof_B, test_B = {}, {}

    for name, fn in [("lgb", fit_lgb_reg), ("xgb", fit_xgb_reg), ("cat", fit_cat_reg)]:
        t = time.time()
        oof_B[name], test_B[name] = fn(X_train, y_rate, X_test, folds)
        r2 = r2_score(y_rate, oof_B[name])
        np.save(OOF_DIR / f"v4_{name}_B_oof.npy", oof_B[name])
        np.save(OOF_DIR / f"v4_{name}_B_test.npy", test_B[name])
        print(f"  v4_{name}_B R²={r2:.4f}  ({time.time()-t:.0f}s)")

    # ==================== STAGE 2: multi-seed ====================
    print_header("Stage-2 multi-seed (3 seeds averaged)")
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
    s2_A_oof, s2_A_test = stage2_multi_seed(X_aug_tr, y_tier, X_aug_te, folds, "A")
    print(f"  stage2_A acc={accuracy_score(y_tier, s2_A_oof.argmax(1)):.4f}  ({time.time()-t:.0f}s)")

    t = time.time()
    s2_B_oof, s2_B_test = stage2_multi_seed(X_aug_tr, y_rate, X_aug_te, folds, "B")
    print(f"  stage2_B R²={r2_score(y_rate, s2_B_oof):.4f}  ({time.time()-t:.0f}s)")

    oof_A["stack2"] = s2_A_oof; test_A["stack2"] = s2_A_test
    oof_B["stack2"] = s2_B_oof; test_B["stack2"] = s2_B_test

    # ==================== ENSEMBLE ====================
    print_header("Ensemble")
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
    print_header("iter4 FINAL OOF")
    print(f"  Accuracy       : {acc_final:.4f}  ({A_method})")
    print(f"  R²             : {r2_final:.4f}  ({B_method})")
    print(f"  Combined score : {combined:.4f}")
    print(f"  vs The Lions 0.88175 → gap = {combined - 0.88175:+.4f}")

    with open(OOF_DIR / "iter4_weights.json", "w") as f:
        json.dump({"A_method": A_method, "B_method": B_method,
                   "A_convex": {k: float(v) for k, v in w_A.items()},
                   "B_convex": {k: float(v) for k, v in w_B.items()},
                   "acc": acc_final, "r2": r2_final, "combined": combined,
                   "task_a_scores": {m: float(accuracy_score(y_tier, oof_A[m].argmax(1)))
                                     for m in oof_A},
                   "task_b_scores": {m: float(r2_score(y_rate, oof_B[m]))
                                     for m in oof_B}},
                  f, indent=2)

    # ==================== SUBMISSION ====================
    tier_pred = final_A_test.argmax(1).astype(int)
    rate_pred = clip_rate(final_B_test)
    sub = pd.DataFrame({ID_COL: ids_test.astype(int).to_numpy(),
                        TARGET_A: tier_pred, TARGET_B: rate_pred})
    assert sub[ID_COL].is_unique and len(sub) == 15000
    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"\n  wrote submission.csv  rows={len(sub)}")
    print(f"  tier_dist={sub[TARGET_A].value_counts().to_dict()}")
    print(f"  rate range=[{sub[TARGET_B].min():.2f}, {sub[TARGET_B].max():.2f}]")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
