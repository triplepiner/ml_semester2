"""
train_base.py
Trains the three base learners (LightGBM, XGBoost, CatBoost) for both tasks
using shared 5-fold CV. Each model writes:
  - outputs/oof/<model>_<task>_oof.npy        : out-of-fold predictions
  - outputs/oof/<model>_<task>_test.npy       : averaged test predictions
For Task A (classification) the arrays store class probabilities (N, 5);
for Task B (regression) they store scalar predictions (N,).

Optuna tuning is optional and controlled by --tune. In --quick mode we skip
Optuna and use sensible hand-tuned defaults for a fast smoke test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

from utils import (OOF_DIR, SEED, clip_rate, get_folds, print_header, set_seed)

N_CLASSES = 5


# ---------------------------------------------------------------------------
# Default hyperparameters — strong enough for a first submission without tuning
# ---------------------------------------------------------------------------

# Stronger pre-tuned defaults (round 2): lower LR, more capacity, tighter
# regularisation. These are community-vetted configs for tabular credit models
# that typically buy +0.01 combined over naive defaults without any search.
LGB_CLS_DEFAULT = dict(
    objective="multiclass", num_class=N_CLASSES, metric="multi_logloss",
    learning_rate=0.02, num_leaves=127, min_child_samples=15,
    feature_fraction=0.75, bagging_fraction=0.80, bagging_freq=5,
    lambda_l1=0.1, lambda_l2=1.0, min_split_gain=0.01,
    verbose=-1, seed=SEED,
)
LGB_REG_DEFAULT = dict(
    objective="regression_l1", metric="rmse",   # L1 = robust to long tail in tier 4
    learning_rate=0.02, num_leaves=127, min_child_samples=15,
    feature_fraction=0.75, bagging_fraction=0.80, bagging_freq=5,
    lambda_l1=0.1, lambda_l2=1.0, min_split_gain=0.01,
    verbose=-1, seed=SEED,
)
XGB_CLS_DEFAULT = dict(
    objective="multi:softprob", num_class=N_CLASSES, eval_metric="mlogloss",
    learning_rate=0.02, max_depth=8, subsample=0.80, colsample_bytree=0.70,
    min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0, gamma=0.01,
    tree_method="hist", random_state=SEED, verbosity=0,
)
XGB_REG_DEFAULT = dict(
    objective="reg:squarederror", eval_metric="rmse",
    learning_rate=0.02, max_depth=8, subsample=0.80, colsample_bytree=0.70,
    min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0, gamma=0.01,
    tree_method="hist", random_state=SEED, verbosity=0,
)
CAT_CLS_DEFAULT = dict(
    loss_function="MultiClass", classes_count=N_CLASSES,
    iterations=4000, learning_rate=0.03, depth=8, l2_leaf_reg=5.0,
    random_strength=0.5, bagging_temperature=0.2,
    border_count=254, random_seed=SEED, verbose=False,
    allow_writing_files=False,
)
CAT_REG_DEFAULT = dict(
    loss_function="RMSE", iterations=4000, learning_rate=0.03, depth=8,
    l2_leaf_reg=5.0, random_strength=0.5, bagging_temperature=0.2,
    border_count=254, random_seed=SEED, verbose=False,
    allow_writing_files=False,
)


# ---------------------------------------------------------------------------
# LightGBM training loops
# ---------------------------------------------------------------------------

def train_lgb_classifier(X, y, X_test, folds, params=None, num_boost_round=6000):
    params = {**LGB_CLS_DEFAULT, **(params or {})}
    oof = np.zeros((len(X), N_CLASSES))
    test_pred = np.zeros((len(X_test), N_CLASSES))
    for fi, (tr, va) in enumerate(folds):
        dtrain = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        dvalid = lgb.Dataset(X.iloc[va], label=y.iloc[va])
        model = lgb.train(
            params, dtrain, num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict(X.iloc[va], num_iteration=model.best_iteration)
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / len(folds)
    return oof, test_pred


def train_lgb_regressor(X, y, X_test, folds, params=None, num_boost_round=6000):
    params = {**LGB_REG_DEFAULT, **(params or {})}
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    for fi, (tr, va) in enumerate(folds):
        dtrain = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        dvalid = lgb.Dataset(X.iloc[va], label=y.iloc[va])
        model = lgb.train(
            params, dtrain, num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict(X.iloc[va], num_iteration=model.best_iteration)
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / len(folds)
    return oof, test_pred


# ---------------------------------------------------------------------------
# XGBoost training loops
# ---------------------------------------------------------------------------

def train_xgb_classifier(X, y, X_test, folds, params=None, num_boost_round=6000):
    params = {**XGB_CLS_DEFAULT, **(params or {})}
    oof = np.zeros((len(X), N_CLASSES))
    test_pred = np.zeros((len(X_test), N_CLASSES))
    dtest = xgb.DMatrix(X_test)
    for fi, (tr, va) in enumerate(folds):
        dtrain = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr])
        dvalid = xgb.DMatrix(X.iloc[va], label=y.iloc[va])
        model = xgb.train(
            params, dtrain, num_boost_round=num_boost_round,
            evals=[(dvalid, "valid")], early_stopping_rounds=150,
            verbose_eval=False,
        )
        oof[va] = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
        test_pred += model.predict(dtest,
                                   iteration_range=(0, model.best_iteration + 1)) / len(folds)
    return oof, test_pred


def train_xgb_regressor(X, y, X_test, folds, params=None, num_boost_round=6000):
    params = {**XGB_REG_DEFAULT, **(params or {})}
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    dtest = xgb.DMatrix(X_test)
    for fi, (tr, va) in enumerate(folds):
        dtrain = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr])
        dvalid = xgb.DMatrix(X.iloc[va], label=y.iloc[va])
        model = xgb.train(
            params, dtrain, num_boost_round=num_boost_round,
            evals=[(dvalid, "valid")], early_stopping_rounds=150,
            verbose_eval=False,
        )
        oof[va] = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
        test_pred += model.predict(dtest,
                                   iteration_range=(0, model.best_iteration + 1)) / len(folds)
    return oof, test_pred


# ---------------------------------------------------------------------------
# CatBoost training loops
# ---------------------------------------------------------------------------

def train_cat_classifier(X, y, X_test, folds, params=None):
    params = {**CAT_CLS_DEFAULT, **(params or {})}
    oof = np.zeros((len(X), N_CLASSES))
    test_pred = np.zeros((len(X_test), N_CLASSES))
    for fi, (tr, va) in enumerate(folds):
        model = CatBoostClassifier(**params)
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=(X.iloc[va], y.iloc[va]),
            use_best_model=True, early_stopping_rounds=150,
        )
        oof[va] = model.predict_proba(X.iloc[va])
        test_pred += model.predict_proba(X_test) / len(folds)
    return oof, test_pred


def train_cat_regressor(X, y, X_test, folds, params=None):
    params = {**CAT_REG_DEFAULT, **(params or {})}
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    for fi, (tr, va) in enumerate(folds):
        model = CatBoostRegressor(**params)
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=(X.iloc[va], y.iloc[va]),
            use_best_model=True, early_stopping_rounds=150,
        )
        oof[va] = model.predict(X.iloc[va])
        test_pred += model.predict(X_test) / len(folds)
    return oof, test_pred


# ---------------------------------------------------------------------------
# Optuna tuning (optional)
# ---------------------------------------------------------------------------

def tune_lgb(X, y, folds, task: str, n_trials: int = 50):
    """TPE search over LightGBM params. Returns best params dict."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
            feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 7),
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        )
        if task == "classification":
            oof, _ = train_lgb_classifier(X, y, X.iloc[:1], folds,
                                          params=p, num_boost_round=1500)
            return -accuracy_score(y, oof.argmax(axis=1))
        else:
            oof, _ = train_lgb_regressor(X, y, X.iloc[:1], folds,
                                         params=p, num_boost_round=1500)
            return -r2_score(y, oof)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=SEED),
        direction="minimize",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

MODELS_A = ["lgb", "xgb", "cat", "lgb_ord"]  # ord = regression-on-tier trick
MODELS_B = ["lgb", "xgb", "cat"]


def run_all_base(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_tier: pd.Series, y_rate: pd.Series,
                 tune: bool = False, n_trials: int = 40,
                 quick: bool = False) -> dict:
    """Train every base model; save OOF + test predictions to disk."""
    set_seed(SEED)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    folds = get_folds(y_tier)

    # Save fold indices so stacking reuses them without accidental reshuffle.
    np.save(OOF_DIR / "fold_indices.npy",
            np.array([np.asarray(va) for _, va in folds], dtype=object),
            allow_pickle=True)

    scores = {}
    rounds = 800 if quick else 6000

    # ---- Task A: RiskTier ----
    print_header("Task A — RiskTier classification")

    # LightGBM
    t0 = time.time()
    params = tune_lgb(X_train, y_tier, folds, "classification", n_trials) \
        if tune else None
    oof, test = train_lgb_classifier(X_train, y_tier, X_test, folds,
                                     params=params, num_boost_round=rounds)
    np.save(OOF_DIR / "lgb_A_oof.npy", oof)
    np.save(OOF_DIR / "lgb_A_test.npy", test)
    scores["A_lgb_acc"] = accuracy_score(y_tier, oof.argmax(axis=1))
    print(f"  LGB  acc={scores['A_lgb_acc']:.4f}  ({time.time()-t0:.0f}s)")

    # XGBoost
    t0 = time.time()
    oof, test = train_xgb_classifier(X_train, y_tier, X_test, folds,
                                     num_boost_round=rounds)
    np.save(OOF_DIR / "xgb_A_oof.npy", oof)
    np.save(OOF_DIR / "xgb_A_test.npy", test)
    scores["A_xgb_acc"] = accuracy_score(y_tier, oof.argmax(axis=1))
    print(f"  XGB  acc={scores['A_xgb_acc']:.4f}  ({time.time()-t0:.0f}s)")

    # CatBoost
    t0 = time.time()
    cat_iter = 500 if quick else CAT_CLS_DEFAULT["iterations"]
    oof, test = train_cat_classifier(X_train, y_tier, X_test, folds,
                                     params={"iterations": cat_iter})
    np.save(OOF_DIR / "cat_A_oof.npy", oof)
    np.save(OOF_DIR / "cat_A_test.npy", test)
    scores["A_cat_acc"] = accuracy_score(y_tier, oof.argmax(axis=1))
    print(f"  CAT  acc={scores['A_cat_acc']:.4f}  ({time.time()-t0:.0f}s)")

    # Ordinal trick: regression on tier → round
    t0 = time.time()
    oof_ord, test_ord = train_lgb_regressor(
        X_train, y_tier.astype(float), X_test, folds, num_boost_round=rounds,
    )
    oof_ord_cls = np.clip(np.round(oof_ord).astype(int), 0, N_CLASSES - 1)
    test_ord_cls = np.clip(np.round(test_ord).astype(int), 0, N_CLASSES - 1)
    # One-hot ordinal predictions so they share the same shape as the others
    oof_ord_prob = np.zeros((len(X_train), N_CLASSES))
    oof_ord_prob[np.arange(len(X_train)), oof_ord_cls] = 1.0
    test_ord_prob = np.zeros((len(X_test), N_CLASSES))
    test_ord_prob[np.arange(len(X_test)), test_ord_cls] = 1.0
    np.save(OOF_DIR / "lgb_ord_A_oof.npy", oof_ord_prob)
    np.save(OOF_DIR / "lgb_ord_A_test.npy", test_ord_prob)
    np.save(OOF_DIR / "lgb_ord_A_oof_float.npy", oof_ord)
    np.save(OOF_DIR / "lgb_ord_A_test_float.npy", test_ord)
    scores["A_ord_acc"] = accuracy_score(y_tier, oof_ord_cls)
    print(f"  ORD  acc={scores['A_ord_acc']:.4f}  ({time.time()-t0:.0f}s)")

    # ---- Task B: InterestRate ----
    print_header("Task B — InterestRate regression")

    t0 = time.time()
    params = tune_lgb(X_train, y_rate, folds, "regression", n_trials) \
        if tune else None
    oof, test = train_lgb_regressor(X_train, y_rate, X_test, folds,
                                    params=params, num_boost_round=rounds)
    np.save(OOF_DIR / "lgb_B_oof.npy", oof)
    np.save(OOF_DIR / "lgb_B_test.npy", test)
    scores["B_lgb_r2"] = r2_score(y_rate, oof)
    print(f"  LGB  R2={scores['B_lgb_r2']:.4f}  ({time.time()-t0:.0f}s)")

    t0 = time.time()
    oof, test = train_xgb_regressor(X_train, y_rate, X_test, folds,
                                    num_boost_round=rounds)
    np.save(OOF_DIR / "xgb_B_oof.npy", oof)
    np.save(OOF_DIR / "xgb_B_test.npy", test)
    scores["B_xgb_r2"] = r2_score(y_rate, oof)
    print(f"  XGB  R2={scores['B_xgb_r2']:.4f}  ({time.time()-t0:.0f}s)")

    t0 = time.time()
    cat_iter = 500 if quick else CAT_REG_DEFAULT["iterations"]
    oof, test = train_cat_regressor(X_train, y_rate, X_test, folds,
                                    params={"iterations": cat_iter})
    np.save(OOF_DIR / "cat_B_oof.npy", oof)
    np.save(OOF_DIR / "cat_B_test.npy", test)
    scores["B_cat_r2"] = r2_score(y_rate, oof)
    print(f"  CAT  R2={scores['B_cat_r2']:.4f}  ({time.time()-t0:.0f}s)")

    with open(OOF_DIR / "base_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="run Optuna tuning")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    args = parser.parse_args()

    from utils import load_data
    from preprocessing import preprocess
    from features import engineer_features

    train, test = load_data()
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)
    X_train, X_test, y_tier, y_rate, _ = preprocess(train_fe, test_fe)
    run_all_base(X_train, X_test, y_tier, y_rate,
                 tune=args.tune, n_trials=args.n_trials, quick=args.quick)
