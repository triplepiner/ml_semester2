"""
stack.py
Stage-2: cross-task stacking + weighted ensembling.

The key insight: RiskTier and InterestRate are nearly collinear in real
lending, so each task's OOF predictions contain information the other's
feature matrix lacks. We build an augmented feature set = original features +
every base model's OOF and train a second-stage LightGBM per task. We then
blend stage-2 with the best stage-1 model via convex weights optimised on OOF.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, r2_score

import lightgbm as lgb

from utils import (N_FOLDS, OOF_DIR, SEED, clip_rate, get_folds, print_header,
                   set_seed)

N_CLASSES = 5


def _load_oof_task(task: str):
    """Return a dict model_name → (oof_array, test_array)."""
    models = {
        "A": ["lgb", "xgb", "cat", "lgb_ord"],
        "B": ["lgb", "xgb", "cat"],
    }[task]
    oofs, tests = {}, {}
    for m in models:
        oofs[m] = np.load(OOF_DIR / f"{m}_{task}_oof.npy")
        tests[m] = np.load(OOF_DIR / f"{m}_{task}_test.npy")
    return oofs, tests


def build_stack_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Concat original features with every base model's OOF predictions."""
    oof_A, test_A = _load_oof_task("A")
    oof_B, test_B = _load_oof_task("B")

    parts_train = [X_train.reset_index(drop=True)]
    parts_test = [X_test.reset_index(drop=True)]

    for m, arr in oof_A.items():
        cols = [f"oofA_{m}_p{k}" for k in range(arr.shape[1])]
        parts_train.append(pd.DataFrame(arr, columns=cols))
        parts_test.append(pd.DataFrame(test_A[m], columns=cols))

    # Include the ordinal regressor's continuous prediction too — it carries
    # more information than the rounded class label.
    ord_oof_float = np.load(OOF_DIR / "lgb_ord_A_oof_float.npy")
    ord_test_float = np.load(OOF_DIR / "lgb_ord_A_test_float.npy")
    parts_train.append(pd.DataFrame({"oofA_lgb_ord_float": ord_oof_float}))
    parts_test.append(pd.DataFrame({"oofA_lgb_ord_float": ord_test_float}))

    for m, arr in oof_B.items():
        parts_train.append(pd.DataFrame({f"oofB_{m}": arr}))
        parts_test.append(pd.DataFrame({f"oofB_{m}": test_B[m]}))

    train_aug = pd.concat(parts_train, axis=1)
    test_aug = pd.concat(parts_test, axis=1)
    return train_aug, test_aug


def train_stage2(X_aug_train, y, task: str, folds, quick: bool = False):
    """Second-stage LightGBM on augmented features."""
    rounds = 400 if quick else 2000
    params = dict(
        learning_rate=0.03, num_leaves=31, min_child_samples=30,
        feature_fraction=0.8, bagging_fraction=0.85, bagging_freq=5,
        lambda_l2=1.0, verbose=-1, seed=SEED,
    )
    if task == "A":
        params.update(objective="multiclass", num_class=N_CLASSES,
                      metric="multi_logloss")
    else:
        params.update(objective="regression", metric="rmse")

    if task == "A":
        oof = np.zeros((len(X_aug_train), N_CLASSES))
    else:
        oof = np.zeros(len(X_aug_train))

    for tr, va in folds:
        dtrain = lgb.Dataset(X_aug_train.iloc[tr], label=y.iloc[tr])
        dvalid = lgb.Dataset(X_aug_train.iloc[va], label=y.iloc[va])
        model = lgb.train(
            params, dtrain, num_boost_round=rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict(X_aug_train.iloc[va],
                                num_iteration=model.best_iteration)
    return oof, params


def fit_stage2_test(X_aug_train, y, X_aug_test, task: str, folds,
                    params: dict, quick: bool = False):
    """Averaged test predictions from stage-2, retraining per fold for stability."""
    rounds = 400 if quick else 2000
    if task == "A":
        test_pred = np.zeros((len(X_aug_test), N_CLASSES))
    else:
        test_pred = np.zeros(len(X_aug_test))

    for tr, va in folds:
        dtrain = lgb.Dataset(X_aug_train.iloc[tr], label=y.iloc[tr])
        dvalid = lgb.Dataset(X_aug_train.iloc[va], label=y.iloc[va])
        model = lgb.train(
            params, dtrain, num_boost_round=rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
        )
        test_pred += model.predict(X_aug_test,
                                   num_iteration=model.best_iteration) / len(folds)
    return test_pred


def optimise_blend_regression(oof_dict: dict, y: np.ndarray) -> np.ndarray:
    """Convex weights that maximise R² on OOF."""
    names = list(oof_dict.keys())
    preds = np.column_stack([oof_dict[n] for n in names])

    def neg_r2(w):
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            return 0.0
        w = w / w.sum()
        return -r2_score(y, preds @ w)

    w0 = np.ones(len(names)) / len(names)
    res = minimize(neg_r2, w0, method="Nelder-Mead",
                   options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-5})
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    return dict(zip(names, w))


def optimise_blend_classification(oof_dict: dict, y: np.ndarray) -> dict:
    """
    Convex weights that maximise accuracy on OOF.

    Accuracy is piecewise-constant in weights, so direct optimisation has flat
    plateaus. We use a two-stage search:
      1. Nelder-Mead on log-loss (smooth) to find a strong neighbourhood.
      2. Small random-restart refinement that directly checks accuracy.
    This reliably beats the coarse grid used previously, which often returned
    equal weights because of plateau ties.
    """
    from scipy.optimize import minimize
    names = list(oof_dict.keys())
    probs = np.stack([oof_dict[n] for n in names])  # (M, N, C)
    M = len(names)
    N, C = probs.shape[1], probs.shape[2]
    eps = 1e-12

    def _normalise(w):
        w = np.clip(w, 0, None)
        s = w.sum()
        return w / s if s > 0 else np.ones(M) / M

    def neg_logloss(w):
        w = _normalise(w)
        blend = (w[:, None, None] * probs).sum(axis=0)
        blend = np.clip(blend, eps, 1 - eps)
        idx = np.arange(N)
        return -np.log(blend[idx, y]).mean()

    # Stage 1 — smooth optimisation
    rng = np.random.default_rng(42)
    best = minimize(neg_logloss, np.ones(M) / M, method="Nelder-Mead",
                    options={"maxiter": 1000, "xatol": 1e-5, "fatol": 1e-7})
    # Try random restarts for robustness
    for _ in range(8):
        w0 = rng.dirichlet(np.ones(M))
        res = minimize(neg_logloss, w0, method="Nelder-Mead",
                       options={"maxiter": 800})
        if res.fun < best.fun:
            best = res
    w_smooth = _normalise(best.x)

    # Stage 2 — direct accuracy refinement via small perturbations
    def acc_of(w):
        blend = (w[:, None, None] * probs).sum(axis=0)
        return accuracy_score(y, blend.argmax(axis=1))

    best_w = w_smooth
    best_acc = acc_of(best_w)
    # Simple coordinate-descent / perturbation
    for _ in range(400):
        w_try = best_w + rng.normal(0, 0.03, size=M)
        w_try = _normalise(w_try)
        a = acc_of(w_try)
        if a > best_acc:
            best_acc, best_w = a, w_try
    # Also compare to equal weights and single-model-best as safety nets
    candidates = [(best_w, best_acc), (np.ones(M) / M, acc_of(np.ones(M) / M))]
    for i in range(M):
        w = np.zeros(M); w[i] = 1.0
        candidates.append((w, acc_of(w)))
    best_w, _ = max(candidates, key=lambda x: x[1])
    return dict(zip(names, best_w))


def run_stacking(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_tier: pd.Series, y_rate: pd.Series,
                 quick: bool = False) -> dict:
    """Full stage-2 pipeline. Returns final OOF + test predictions for each task."""
    set_seed(SEED)
    print_header("Stage-2: cross-task stacking")

    folds = get_folds(y_tier)
    X_aug_train, X_aug_test = build_stack_features(X_train, X_test)

    # Stage-2 LGBM per task
    oof_A_stack, params_A = train_stage2(X_aug_train, y_tier, "A", folds, quick=quick)
    test_A_stack = fit_stage2_test(X_aug_train, y_tier, X_aug_test, "A",
                                   folds, params_A, quick=quick)
    acc_stack = accuracy_score(y_tier, oof_A_stack.argmax(axis=1))
    print(f"  Stage-2 A  acc={acc_stack:.4f}")

    oof_B_stack, params_B = train_stage2(X_aug_train, y_rate, "B", folds, quick=quick)
    test_B_stack = fit_stage2_test(X_aug_train, y_rate, X_aug_test, "B",
                                   folds, params_B, quick=quick)
    r2_stack = r2_score(y_rate, oof_B_stack)
    print(f"  Stage-2 B   R²={r2_stack:.4f}")

    np.save(OOF_DIR / "stack2_A_oof.npy", oof_A_stack)
    np.save(OOF_DIR / "stack2_A_test.npy", test_A_stack)
    np.save(OOF_DIR / "stack2_B_oof.npy", oof_B_stack)
    np.save(OOF_DIR / "stack2_B_test.npy", test_B_stack)

    # ---- Final blend: stage-2 + best stage-1 models ----
    oof_A, test_A = _load_oof_task("A")
    oof_A["stack2"] = oof_A_stack
    test_A["stack2"] = test_A_stack

    oof_B, test_B = _load_oof_task("B")
    oof_B["stack2"] = oof_B_stack
    test_B["stack2"] = test_B_stack

    w_A = optimise_blend_classification(oof_A, y_tier.to_numpy())
    w_B = optimise_blend_regression(oof_B, y_rate.to_numpy())

    # Apply weights
    final_oof_A = sum(w_A[n] * oof_A[n] for n in w_A)
    final_test_A = sum(w_A[n] * test_A[n] for n in w_A)
    final_oof_B = sum(w_B[n] * oof_B[n] for n in w_B)
    final_test_B = sum(w_B[n] * test_B[n] for n in w_B)

    acc_final = accuracy_score(y_tier, final_oof_A.argmax(axis=1))
    r2_final = r2_score(y_rate, final_oof_B)
    combined = 0.5 * acc_final + 0.5 * r2_final

    print_header("Final ensemble (OOF estimate of leaderboard)")
    print(f"  Accuracy       : {acc_final:.4f}")
    print(f"  R²             : {r2_final:.4f}")
    print(f"  Combined score : {combined:.4f}")
    print(f"  Task A weights : {w_A}")
    print(f"  Task B weights : {w_B}")

    np.save(OOF_DIR / "final_A_oof.npy", final_oof_A)
    np.save(OOF_DIR / "final_A_test.npy", final_test_A)
    np.save(OOF_DIR / "final_B_oof.npy", final_oof_B)
    np.save(OOF_DIR / "final_B_test.npy", final_test_B)

    with open(OOF_DIR / "final_weights.json", "w") as f:
        json.dump({"A": {k: float(v) for k, v in w_A.items()},
                   "B": {k: float(v) for k, v in w_B.items()},
                   "acc": acc_final, "r2": r2_final,
                   "combined": combined}, f, indent=2)

    return {"acc": acc_final, "r2": r2_final, "combined": combined,
            "w_A": w_A, "w_B": w_B,
            "final_test_A": final_test_A, "final_test_B": final_test_B}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    from utils import load_data
    from preprocessing import preprocess
    from features import engineer_features

    train, test = load_data()
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)
    X_train, X_test, y_tier, y_rate, _ = preprocess(train_fe, test_fe)
    run_stacking(X_train, X_test, y_tier, y_rate, quick=args.quick)
