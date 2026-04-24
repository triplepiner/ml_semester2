"""
reproduce_final.py — Single-entry pipeline that reproduces the submitted
Kaggle CSV from the raw data files in ../data/.

Team: Slavs  (Makar Ulesov, Ivan Kanev, Delyan Hristov)
Course: AI1215 Introduction to Machine Learning (Spring 2026)

Usage
-----
    cd code/
    python reproduce_final.py

Expected OOF combined ≈ 0.8407 (0.5 · Accuracy + 0.5 · R²).
Expected runtime 60-80 minutes on an 8-core CPU.
All randomness is seeded via utils.set_seed (SEED=42).

Pipeline outline
----------------
Phase 1  — Load + engineer features + preprocess + multi-target encoding
Phase 2  — Base learners at LR=0.02 for both tasks:
             LGB, XGB, CatBoost classifiers for RiskTier
             LGB, XGB, CatBoost regressors for InterestRate
             LGB regression-on-tier (ordinal trick)
             Two-stage Task A (binary is_tier4 + 4-class)
Phase 3  — Initial ensemble → pseudo-label top 5000 high-confidence test rows
Phase 4  — Retrain LGB / XGB / CatBoost on enlarged training (35 000 + 5 000)
Phase 5  — Compute K-nearest-neighbor target features
Phase 6  — Stage-2 multi-seed LightGBM meta-learner
Phase 7  — Blend candidates (convex vs Ridge), pick the higher-OOF winner
Phase 8  — Validate + write outputs/submission.csv

See written report Section 3 for an ablation table showing which techniques
actually moved the score and which did not.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

# Shared utilities
from utils import (ID_COL, OOF_DIR, OUT_DIR, RATE_MAX, RATE_MIN, SEED,
                   TARGET_A, TARGET_B, clip_rate, get_folds, load_data,
                   print_header, set_seed, KFoldTargetEncoder)
from preprocessing import preprocess
from features import engineer_features

N_CLASSES = 5
PSEUDO_MAX_ROWS = 5000
PSEUDO_THRESHOLD = 0.90
K_NEIGHBORS = 10


# ---------------------------------------------------------------------------
# Multi-target encoding (mean rate + P(tier=4) + per-group std). The same
# K-fold scheme is used for train so encodings can't leak across folds.
# ---------------------------------------------------------------------------

def multi_target_encode(train_raw, test_raw, cols, y_tier, y_rate,
                        n_splits: int = 5, smoothing: float = 20.0,
                        seed: int = SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    enc_tr = pd.DataFrame(index=train_raw.index)
    enc_te = pd.DataFrame(index=test_raw.index)
    gmean = float(y_rate.mean())
    gt4 = float((y_tier == 4).mean())
    gstd = float(y_rate.std())
    is_tier4 = (y_tier == 4).astype(int)

    for c in cols:
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

        agg_r = y_rate.groupby(train_raw[c]).agg(["sum", "count", "std"])
        sm_mean = (agg_r["sum"] + smoothing * gmean) / (agg_r["count"] + smoothing)
        sm_std = agg_r["std"].fillna(gstd)
        agg_t = is_tier4.groupby(train_raw[c]).agg(["sum", "count"])
        sm_p4 = (agg_t["sum"] + smoothing * gt4) / (agg_t["count"] + smoothing)
        enc_te[f"{c}_te_rate"] = test_raw[c].map(sm_mean).fillna(gmean).astype(float)
        enc_te[f"{c}_te_p4"] = test_raw[c].map(sm_p4).fillna(gt4).astype(float)
        enc_te[f"{c}_te_std"] = test_raw[c].map(sm_std).fillna(gstd).astype(float)
    return enc_tr, enc_te


# ---------------------------------------------------------------------------
# Base learners (LightGBM / XGBoost / CatBoost)
# ---------------------------------------------------------------------------

def fit_lgb_cls(X, y, Xt, folds, rounds: int = 6000):
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


def fit_xgb_cls(X, y, Xt, folds, rounds: int = 6000):
    p = dict(objective="multi:softprob", num_class=N_CLASSES, eval_metric="mlogloss",
             learning_rate=0.02, max_depth=8, subsample=0.80, colsample_bytree=0.70,
             min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0, gamma=0.01,
             tree_method="hist", random_state=SEED, verbosity=0)
    oof = np.zeros((len(X), N_CLASSES)); tp = np.zeros((len(Xt), N_CLASSES))
    dt = xgb.DMatrix(Xt)
    for tri, vai in folds:
        d_tr = xgb.DMatrix(X.iloc[tri], label=y.iloc[tri])
        d_va = xgb.DMatrix(X.iloc[vai], label=y.iloc[vai])
        m = xgb.train(p, d_tr, rounds, [(d_va, "v")],
                      early_stopping_rounds=200, verbose_eval=False)
        oof[vai] = m.predict(d_va, iteration_range=(0, m.best_iteration + 1))
        tp += m.predict(dt, iteration_range=(0, m.best_iteration + 1)) / len(folds)
    return oof, tp


def fit_cat_cls(X, y, Xt, folds, iters: int = 4000):
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


def fit_lgb_reg(X, y, Xt, folds, rounds: int = 6000):
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


def fit_xgb_reg(X, y, Xt, folds, rounds: int = 6000):
    p = dict(objective="reg:squarederror", eval_metric="rmse",
             learning_rate=0.02, max_depth=8, subsample=0.80,
             colsample_bytree=0.70, min_child_weight=3, reg_alpha=0.1,
             reg_lambda=1.0, gamma=0.01, tree_method="hist",
             random_state=SEED, verbosity=0)
    oof = np.zeros(len(X)); tp = np.zeros(len(Xt))
    dt = xgb.DMatrix(Xt)
    for tri, vai in folds:
        d_tr = xgb.DMatrix(X.iloc[tri], label=y.iloc[tri])
        d_va = xgb.DMatrix(X.iloc[vai], label=y.iloc[vai])
        m = xgb.train(p, d_tr, rounds, [(d_va, "v")],
                      early_stopping_rounds=200, verbose_eval=False)
        oof[vai] = m.predict(d_va, iteration_range=(0, m.best_iteration + 1))
        tp += m.predict(dt, iteration_range=(0, m.best_iteration + 1)) / len(folds)
    return oof, tp


def fit_cat_reg(X, y, Xt, folds, iters: int = 4000):
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


def fit_ordinal_lgb(X, y_tier, Xt, folds, rounds: int = 3000):
    """Task A trick: treat tier as float, regress, round to class at inference."""
    y_float = y_tier.astype(float)
    oof_f, tp_f = fit_lgb_reg(X, y_float, Xt, folds, rounds)
    oof_cls = np.clip(np.round(oof_f).astype(int), 0, N_CLASSES - 1)
    tp_cls = np.clip(np.round(tp_f).astype(int), 0, N_CLASSES - 1)
    oof_prob = np.eye(N_CLASSES)[oof_cls]
    tp_prob = np.eye(N_CLASSES)[tp_cls]
    return oof_prob, tp_prob, oof_f, tp_f


def fit_two_stage_tier(X, y_tier, Xt, folds, rounds: int = 3500):
    """Binary (is_tier4) → 4-class on non-tier-4. Combines via product of probs."""
    is_4 = (y_tier == 4).astype(int)
    oof = np.zeros((len(X), N_CLASSES)); tp = np.zeros((len(Xt), N_CLASSES))
    for tri, vai in folds:
        # Binary head
        bin_p = dict(objective="binary", metric="auc", learning_rate=0.03,
                     num_leaves=127, min_child_samples=15, feature_fraction=0.75,
                     bagging_fraction=0.80, bagging_freq=5, lambda_l2=1.0,
                     verbose=-1, seed=SEED)
        bin_m = lgb.train(bin_p, lgb.Dataset(X.iloc[tri], is_4.iloc[tri]), rounds,
                          valid_sets=[lgb.Dataset(X.iloc[vai], is_4.iloc[vai])],
                          callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        p4_va = bin_m.predict(X.iloc[vai], num_iteration=bin_m.best_iteration)
        p4_te = bin_m.predict(Xt, num_iteration=bin_m.best_iteration)

        # 4-class head on non-tier-4 train rows
        prime_idx = np.array([i for i in tri if y_tier.iloc[i] != 4])
        y_prime = y_tier.iloc[prime_idx].to_numpy()
        m4_p = dict(objective="multiclass", num_class=4, metric="multi_logloss",
                    learning_rate=0.02, num_leaves=127, min_child_samples=15,
                    feature_fraction=0.75, bagging_fraction=0.80, bagging_freq=5,
                    lambda_l2=1.0, verbose=-1, seed=SEED)
        val_prime = np.array([i for i in vai if y_tier.iloc[i] != 4])
        if len(val_prime) < 50:
            val_prime = vai
            val_labels = y_tier.iloc[vai].clip(0, 3).to_numpy()
        else:
            val_labels = y_tier.iloc[val_prime].to_numpy()
        m4 = lgb.train(m4_p, lgb.Dataset(X.iloc[prime_idx], y_prime), rounds,
                       valid_sets=[lgb.Dataset(X.iloc[val_prime], val_labels)],
                       callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        p03_va = m4.predict(X.iloc[vai], num_iteration=m4.best_iteration)
        p03_te = m4.predict(Xt, num_iteration=m4.best_iteration)

        # Combine
        oof[vai, 0:4] = (1 - p4_va)[:, None] * p03_va
        oof[vai, 4] = p4_va
        tp[:, 0:4] += ((1 - p4_te)[:, None] * p03_te) / len(folds)
        tp[:, 4] += p4_te / len(folds)
    return oof, tp


# ---------------------------------------------------------------------------
# K-Nearest-Neighbor target features
# ---------------------------------------------------------------------------

def knn_target_features(X_train, X_test, y_tier, y_rate, folds, k: int = K_NEIGHBORS):
    """Leave-fold-out KNN: neighbors of train rows are drawn from other folds."""
    std_cols = [c for c in X_train.columns if X_train[c].std() > 1e-6]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train[std_cols].to_numpy())
    X_te_s = scaler.transform(X_test[std_cols].to_numpy())

    knn_mean_rate = np.zeros(len(X_train))
    knn_std_rate = np.zeros(len(X_train))
    knn_p_tier = np.zeros((len(X_train), N_CLASSES))

    for tri, vai in folds:
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        nn.fit(X_tr_s[tri])
        _, idx = nn.kneighbors(X_tr_s[vai])
        neighbour_rates = y_rate.to_numpy()[tri][idx]
        neighbour_tiers = y_tier.to_numpy()[tri][idx]
        knn_mean_rate[vai] = neighbour_rates.mean(axis=1)
        knn_std_rate[vai] = neighbour_rates.std(axis=1)
        for c in range(N_CLASSES):
            knn_p_tier[vai, c] = (neighbour_tiers == c).mean(axis=1)

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(X_tr_s)
    _, idx = nn.kneighbors(X_te_s)
    n_rates = y_rate.to_numpy()[idx]; n_tiers = y_tier.to_numpy()[idx]
    te_mean = n_rates.mean(axis=1); te_std = n_rates.std(axis=1)
    te_p = np.zeros((len(X_test), N_CLASSES))
    for c in range(N_CLASSES):
        te_p[:, c] = (n_tiers == c).mean(axis=1)

    tr_df = pd.DataFrame({"knn_mean_rate": knn_mean_rate,
                          "knn_std_rate": knn_std_rate,
                          **{f"knn_p_tier{c}": knn_p_tier[:, c] for c in range(N_CLASSES)}})
    te_df = pd.DataFrame({"knn_mean_rate": te_mean,
                          "knn_std_rate": te_std,
                          **{f"knn_p_tier{c}": te_p[:, c] for c in range(N_CLASSES)}})
    return tr_df, te_df


# ---------------------------------------------------------------------------
# Stage-2 meta-learner (multi-seed LightGBM)
# ---------------------------------------------------------------------------

def stage2_lgb(X_aug_tr, y, X_aug_te, folds, task: str,
               seeds=(42, 1337, 2024)):
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
            m = lgb.train(p, lgb.Dataset(X_aug_tr.iloc[tri], y.iloc[tri]), 3000,
                          valid_sets=[lgb.Dataset(X_aug_tr.iloc[vai], y.iloc[vai])],
                          callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
            oof[vai] = m.predict(X_aug_tr.iloc[vai], num_iteration=m.best_iteration)
            tp += m.predict(X_aug_te, num_iteration=m.best_iteration) / len(folds)
        oof_all += oof / len(seeds); test_all += tp / len(seeds)
    return oof_all, test_all


# ---------------------------------------------------------------------------
# Ensemble blenders
# ---------------------------------------------------------------------------

def blend_convex_cls(oof_dict, y):
    """Weighted blend of class probabilities, weights in the simplex."""
    names = list(oof_dict.keys())
    probs = np.stack([oof_dict[n] for n in names])
    M = len(names); eps = 1e-12

    def _norm(w):
        w = np.clip(w, 0, None); s = w.sum()
        return w / s if s > 0 else np.ones(M) / M

    def neg_logloss(w):
        w = _norm(w)
        blend = (w[:, None, None] * probs).sum(axis=0)
        blend = np.clip(blend, eps, 1 - eps)
        return -np.log(blend[np.arange(len(y)), y]).mean()

    rng = np.random.default_rng(SEED)
    best = minimize(neg_logloss, np.ones(M) / M, method="Nelder-Mead",
                    options={"maxiter": 1000})
    for _ in range(8):
        w0 = rng.dirichlet(np.ones(M))
        res = minimize(neg_logloss, w0, method="Nelder-Mead", options={"maxiter": 800})
        if res.fun < best.fun:
            best = res
    w = _norm(best.x)

    def acc_of(w):
        blend = (w[:, None, None] * probs).sum(axis=0)
        return accuracy_score(y, blend.argmax(axis=1))

    for _ in range(400):
        w_try = _norm(w + rng.normal(0, 0.03, size=M))
        if acc_of(w_try) > acc_of(w):
            w = w_try
    return dict(zip(names, w))


def blend_convex_reg(oof_dict, y):
    names = list(oof_dict.keys())
    P = np.column_stack([oof_dict[n] for n in names])

    def neg_r2(w):
        w = np.clip(w, 0, None); s = w.sum()
        if s == 0: return 0.0
        return -r2_score(y, P @ (w / s))

    res = minimize(neg_r2, np.ones(len(names)) / len(names), method="Nelder-Mead",
                   options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-5})
    w = np.clip(res.x, 0, None); w = w / w.sum()
    return dict(zip(names, w))


def blend_ridge_cls(oof_dict, y, test_dict, alpha: float = 1.0):
    names = list(oof_dict.keys())
    P = np.hstack([oof_dict[n] for n in names])
    Y = np.eye(N_CLASSES)[y]
    m = Ridge(alpha=alpha, random_state=SEED).fit(P, Y)
    oof_pred = m.predict(P)
    Pt = np.hstack([test_dict[n] for n in names])
    test_pred = m.predict(Pt)
    return oof_pred, test_pred


def blend_ridge_reg(oof_dict, y, test_dict, alpha: float = 1.0):
    names = list(oof_dict.keys())
    P = np.column_stack([oof_dict[n] for n in names])
    m = Ridge(alpha=alpha, random_state=SEED).fit(P, y)
    oof_pred = m.predict(P)
    Pt = np.column_stack([test_dict[n] for n in names])
    test_pred = m.predict(Pt)
    return oof_pred, test_pred


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    t0 = time.time()

    # -------- Phase 1: Feature engineering + preprocessing + multi-TE --------
    print_header("Phase 1 — Load + feature engineering + preprocessing")
    train_raw, test_raw = load_data()
    train_fe = engineer_features(train_raw)
    test_fe = engineer_features(test_raw)
    X_train, X_test, y_tier, y_rate, ids_test = preprocess(train_fe, test_fe)

    # Multi-target encoding for high-card categoricals (noise-aware version
    # tuned to keep train/test distributions close).
    te_cols = [c for c in ["State", "JobCategory", "LoanPurpose",
                           "EmployerType", "EmploymentStatus"]
               if c in train_raw.columns]
    if te_cols:
        enc_tr, enc_te = multi_target_encode(
            train_raw[te_cols].fillna("NA").astype(str),
            test_raw[te_cols].fillna("NA").astype(str),
            te_cols, y_tier, y_rate, n_splits=5, smoothing=20)
        X_train = pd.concat([X_train.reset_index(drop=True),
                             enc_tr.reset_index(drop=True)], axis=1)
        X_test = pd.concat([X_test.reset_index(drop=True),
                            enc_te.reset_index(drop=True)], axis=1)
    print(f"  X_train={X_train.shape}  X_test={X_test.shape}")

    folds = get_folds(y_tier)

    # -------- Phase 2: Base learners (LR 0.02, 6000 rounds early-stopped) ----
    print_header("Phase 2 — Base learners at LR 0.02")
    oof_A, test_A = {}, {}
    oof_B, test_B = {}, {}

    for name, fn in [("lgb", fit_lgb_cls), ("xgb", fit_xgb_cls), ("cat", fit_cat_cls)]:
        t = time.time()
        oof_A[name], test_A[name] = fn(X_train, y_tier, X_test, folds)
        acc = accuracy_score(y_tier, oof_A[name].argmax(1))
        print(f"  A/{name:3s} acc={acc:.4f}  ({time.time()-t:.0f}s)")

    for name, fn in [("lgb", fit_lgb_reg), ("xgb", fit_xgb_reg), ("cat", fit_cat_reg)]:
        t = time.time()
        oof_B[name], test_B[name] = fn(X_train, y_rate, X_test, folds)
        r2 = r2_score(y_rate, oof_B[name])
        print(f"  B/{name:3s} R²={r2:.4f}  ({time.time()-t:.0f}s)")

    # Ordinal LGB + two-stage classifier for Task A
    print_header("Phase 2b — Task A specialists")
    t = time.time()
    ord_p, ord_t, ord_f, ord_tf = fit_ordinal_lgb(X_train, y_tier, X_test, folds)
    oof_A["lgb_ord"] = ord_p; test_A["lgb_ord"] = ord_t
    ord_float_train, ord_float_test = ord_f, ord_tf
    print(f"  lgb_ord acc={accuracy_score(y_tier, ord_p.argmax(1)):.4f}  ({time.time()-t:.0f}s)")

    t = time.time()
    oof_A["two_stage"], test_A["two_stage"] = fit_two_stage_tier(
        X_train, y_tier, X_test, folds)
    print(f"  two_stage acc={accuracy_score(y_tier, oof_A['two_stage'].argmax(1)):.4f}  ({time.time()-t:.0f}s)")

    # -------- Phase 3: pseudo-label generation ------------------------------
    print_header("Phase 3 — Initial ensemble → pseudo-labels")
    # Very simple initial blend = average of base probs / predictions
    probs_init_A = np.mean([oof_A[k] for k in oof_A if k != "two_stage"], axis=0)
    probs_init_A /= probs_init_A.sum(axis=1, keepdims=True)
    test_probs_init_A = np.mean([test_A[k] for k in test_A if k != "two_stage"], axis=0)
    test_probs_init_A /= test_probs_init_A.sum(axis=1, keepdims=True)
    test_rate_init = np.mean([test_B[k] for k in test_B], axis=0)

    conf = test_probs_init_A.max(axis=1)
    idx_by_conf = np.argsort(-conf)
    top_idx = idx_by_conf[:PSEUDO_MAX_ROWS]
    top_idx = top_idx[conf[top_idx] >= PSEUDO_THRESHOLD]
    tier_pseudo = test_probs_init_A.argmax(axis=1)
    rate_pseudo = np.clip(test_rate_init, RATE_MIN, RATE_MAX)
    n_real = len(X_train); n_pseudo = len(top_idx)
    print(f"  selected {n_pseudo} pseudo rows (min conf = {conf[top_idx].min():.3f})")

    # -------- Phase 4: retrain on enlarged training -------------------------
    print_header("Phase 4 — Retrain base learners on enlarged training")
    X_enl = pd.concat([X_train.reset_index(drop=True),
                       X_test.iloc[top_idx].reset_index(drop=True)],
                      axis=0, ignore_index=True)
    y_tier_enl = pd.concat([y_tier.reset_index(drop=True),
                            pd.Series(tier_pseudo[top_idx])], axis=0, ignore_index=True)
    y_rate_enl = pd.concat([y_rate.reset_index(drop=True),
                            pd.Series(rate_pseudo[top_idx])], axis=0, ignore_index=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds_enl = list(skf.split(np.arange(len(X_enl)), y_tier_enl))

    oof_A_enl, test_A_enl = {}, {}
    oof_B_enl, test_B_enl = {}, {}
    for name, fn in [("lgb", fit_lgb_cls), ("xgb", fit_xgb_cls), ("cat", fit_cat_cls)]:
        t = time.time()
        oof_A_enl[name], test_A_enl[name] = fn(X_enl, y_tier_enl, X_test, folds_enl)
        print(f"  A/{name:3s} (real OOF acc={accuracy_score(y_tier, oof_A_enl[name][:n_real].argmax(1)):.4f})  ({time.time()-t:.0f}s)")
    for name, fn in [("lgb", fit_lgb_reg), ("xgb", fit_xgb_reg), ("cat", fit_cat_reg)]:
        t = time.time()
        oof_B_enl[name], test_B_enl[name] = fn(X_enl, y_rate_enl, X_test, folds_enl)
        print(f"  B/{name:3s} (real OOF R²={r2_score(y_rate, oof_B_enl[name][:n_real]):.4f})  ({time.time()-t:.0f}s)")

    # Pad the legacy Task A tricks to enlarged length (use their test prediction for pseudo slice)
    ord_pad_oof = np.concatenate([ord_p, ord_t[top_idx]], axis=0)
    ts_pad_oof = np.concatenate([oof_A["two_stage"], test_A["two_stage"][top_idx]], axis=0)
    ord_float_pad = np.concatenate([ord_float_train, ord_float_test[top_idx]], axis=0)

    # Assemble dictionaries used by stage-2 + ensemble
    oof_A_all = {**oof_A_enl, "lgb_ord": ord_pad_oof, "two_stage": ts_pad_oof}
    test_A_all = {**test_A_enl, "lgb_ord": test_A["lgb_ord"], "two_stage": test_A["two_stage"]}
    oof_B_all = {**oof_B_enl}
    test_B_all = {**test_B_enl}

    # -------- Phase 5: KNN target features ----------------------------------
    print_header("Phase 5 — K-NN target features")
    t = time.time()
    knn_tr, knn_te = knn_target_features(X_train, X_test, y_tier, y_rate, folds, k=K_NEIGHBORS)
    # Pad knn_tr for enlarged training: for pseudo rows use neighbours-of-test values
    knn_tr_enl = pd.concat([knn_tr.reset_index(drop=True),
                            knn_te.iloc[top_idx].reset_index(drop=True)],
                           axis=0, ignore_index=True)
    print(f"  KNN features computed ({time.time()-t:.0f}s)")

    # -------- Phase 6: stage-2 multi-seed -----------------------------------
    print_header("Phase 6 — Stage-2 multi-seed LightGBM")
    X_aug_tr = [X_enl.reset_index(drop=True), knn_tr_enl.reset_index(drop=True)]
    X_aug_te = [X_test.reset_index(drop=True), knn_te.reset_index(drop=True)]
    for m, arr in oof_A_all.items():
        cols = [f"oofA_{m}_p{k}" for k in range(arr.shape[1])]
        X_aug_tr.append(pd.DataFrame(arr, columns=cols))
        X_aug_te.append(pd.DataFrame(test_A_all[m], columns=cols))
    X_aug_tr.append(pd.DataFrame({"oofA_ord_float": ord_float_pad}))
    X_aug_te.append(pd.DataFrame({"oofA_ord_float": ord_float_test}))
    for m, arr in oof_B_all.items():
        X_aug_tr.append(pd.DataFrame({f"oofB_{m}": arr}))
        X_aug_te.append(pd.DataFrame({f"oofB_{m}": test_B_all[m]}))
    X_aug_tr = pd.concat(X_aug_tr, axis=1)
    X_aug_te = pd.concat(X_aug_te, axis=1)
    print(f"  X_aug_tr={X_aug_tr.shape}")

    t = time.time()
    s2_A_oof, s2_A_test = stage2_lgb(X_aug_tr, y_tier_enl, X_aug_te, folds_enl, "A")
    acc_s2 = accuracy_score(y_tier, s2_A_oof[:n_real].argmax(1))
    print(f"  stage2_A real-OOF acc={acc_s2:.4f}  ({time.time()-t:.0f}s)")
    t = time.time()
    s2_B_oof, s2_B_test = stage2_lgb(X_aug_tr, y_rate_enl, X_aug_te, folds_enl, "B")
    r2_s2 = r2_score(y_rate, s2_B_oof[:n_real])
    print(f"  stage2_B real-OOF  R²={r2_s2:.4f}  ({time.time()-t:.0f}s)")

    oof_A_all["stack2"] = s2_A_oof; test_A_all["stack2"] = s2_A_test
    oof_B_all["stack2"] = s2_B_oof; test_B_all["stack2"] = s2_B_test

    # -------- Phase 7: ensemble (convex vs Ridge) ---------------------------
    print_header("Phase 7 — Ensemble selection")
    # Slice to real rows for metric evaluation
    oof_A_real = {k: v[:n_real] for k, v in oof_A_all.items()}
    oof_B_real = {k: v[:n_real] for k, v in oof_B_all.items()}

    w_A = blend_convex_cls(oof_A_real, y_tier.to_numpy())
    fA_conv_oof = sum(w_A[n] * oof_A_real[n] for n in w_A)
    fA_conv_test = sum(w_A[n] * test_A_all[n] for n in w_A)
    acc_conv = accuracy_score(y_tier, fA_conv_oof.argmax(1))
    rA_oof, rA_test = blend_ridge_cls(oof_A_real, y_tier.to_numpy(), test_A_all, alpha=1.0)
    acc_ridge = accuracy_score(y_tier, rA_oof.argmax(1))
    print(f"  A convex acc={acc_conv:.4f}   ridge acc={acc_ridge:.4f}")
    if acc_ridge > acc_conv:
        final_A_oof, final_A_test = rA_oof, rA_test; acc_final, A_method = acc_ridge, "ridge"
    else:
        final_A_oof, final_A_test = fA_conv_oof, fA_conv_test; acc_final, A_method = acc_conv, "convex"

    w_B = blend_convex_reg(oof_B_real, y_rate.to_numpy())
    fB_conv_oof = sum(w_B[n] * oof_B_real[n] for n in w_B)
    fB_conv_test = sum(w_B[n] * test_B_all[n] for n in w_B)
    r2_conv = r2_score(y_rate, fB_conv_oof)
    rB_oof, rB_test = blend_ridge_reg(oof_B_real, y_rate.to_numpy(), test_B_all, alpha=1.0)
    r2_ridge = r2_score(y_rate, rB_oof)
    print(f"  B convex R²={r2_conv:.4f}   ridge R²={r2_ridge:.4f}")
    if r2_ridge > r2_conv:
        final_B_oof, final_B_test = rB_oof, rB_test; r2_final, B_method = r2_ridge, "ridge"
    else:
        final_B_oof, final_B_test = fB_conv_oof, fB_conv_test; r2_final, B_method = r2_conv, "convex"

    combined = 0.5 * acc_final + 0.5 * r2_final
    print_header("Final OOF estimate")
    print(f"  Accuracy       : {acc_final:.4f}  ({A_method})")
    print(f"  R²             : {r2_final:.4f}  ({B_method})")
    print(f"  Combined score : {combined:.4f}")

    # -------- Phase 8: submission CSV ---------------------------------------
    print_header("Phase 8 — Submission")
    tier_pred = final_A_test.argmax(1).astype(int)
    rate_pred = clip_rate(final_B_test)
    sub = pd.DataFrame({ID_COL: ids_test.astype(int).to_numpy(),
                        TARGET_A: tier_pred, TARGET_B: rate_pred})
    assert sub[ID_COL].is_unique and len(sub) == 15000
    assert set(sub[TARGET_A]).issubset(set(range(N_CLASSES)))
    assert sub[TARGET_B].between(RATE_MIN, RATE_MAX).all()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    # Also persist a machine-readable summary for the report
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    with open(OOF_DIR / "reproduce_final_summary.json", "w") as f:
        json.dump({"acc": acc_final, "r2": r2_final, "combined": combined,
                   "A_method": A_method, "B_method": B_method,
                   "A_weights": {k: float(v) for k, v in w_A.items()},
                   "B_weights": {k: float(v) for k, v in w_B.items()},
                   "n_pseudo": int(n_pseudo),
                   "runtime_minutes": (time.time() - t0) / 60}, f, indent=2)

    print(f"  submission written → {OUT_DIR/'submission.csv'}  rows={len(sub)}")
    print(f"  tier distribution : {sub[TARGET_A].value_counts().to_dict()}")
    print(f"  rate range        : [{sub[TARGET_B].min():.2f}, {sub[TARGET_B].max():.2f}]")
    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
