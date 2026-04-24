"""
advanced.py — Round 3 improvements that sit on top of the base pipeline.

Organised as optional stages that can be toggled on/off:
  - Monotone constraints for LightGBM/XGBoost (domain-knowledge regularisation).
  - Multi-target encoding (mean InterestRate + P(tier=4) + per-group std).
  - Tier-4 specialist regressor — mixture-of-experts for Task B.
  - Two-stage Task A — binary is_tier4 × 4-class tier_0_to_3.
  - DART boosting variant for diversity.
  - Ridge/Lasso meta-blender that allows negative weights.
  - Quantile regression features for Task B stage-2.

Every function is deterministic given SEED=42 and the same fold indices.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

from utils import OOF_DIR, SEED, get_folds, set_seed


N_CLASSES = 5


# ---------------------------------------------------------------------------
# Monotone constraints
# ---------------------------------------------------------------------------

# Feature name → sign ({+1,-1}). + = higher value ⇒ higher target.
# Target semantics differ per task, so we build one dict per (task, family).
# For RISK TIER: + = worse borrower.
MONO_RISK = {
    "NumberOfLatePayments30Days": 1,
    "NumberOfLatePayments60Days": 1,
    "NumberOfLatePayments90Days": 1,
    "NumberOfChargeOffs": 1,
    "NumberOfCollections": 1,
    "NumberOfPublicRecords": 1,
    "NumberOfBankruptcies": 1,
    "NumberOfHardInquiries12Mo": 1,
    "NumberOfHardInquiries24Mo": 1,
    "RevolvingUtilizationRate": 1,
    "DebtToIncomeRatio": 1,
    "LoanToIncomeRatio": 1,
    "PaymentToIncomeRatio": 1,
    "feat_DelinquencyScore": 1,
    "feat_DerogatoryScore": 1,
    "feat_BadEventsTotal": 1,
    "feat_HighUtilization": 1,
    "feat_MaxedOut": 1,
    "feat_AnyDerogatory": 1,
    "feat_SevereDelinquency": 1,
    "AnnualIncome": -1,
    "TotalMonthlyIncome": -1,
    "TotalAssets": -1,
    "SavingsBalance": -1,
    "CheckingBalance": -1,
    "NumberOfSatisfactoryAccounts": -1,
    "feat_LiquidCash": -1,
    "feat_LiquidToLoan": -1,
    "feat_CashRunwayMonths": -1,
    "feat_SatisfactoryRatio": -1,
    "feat_NetWorth": -1,
    "CreditHistoryLengthMonths": -1,
    "OldestAccountAgeMonths": -1,
    "feat_ThinFile": 1,
    "feat_ThickFile": -1,
    "feat_IncomeVerified": -1,
    "feat_IsHomeOwner": -1,
    "feat_HasCollateral": -1,
    "feat_RepeatCustomer": -1,
}

# Rate and tier are co-monotone, so the same mapping applies to Task B.
MONO_RATE = MONO_RISK


def monotone_vector_for(columns: Iterable[str], mapping: dict[str, int]) -> list[int]:
    """Return a list aligned to `columns` with mapping values, 0 elsewhere."""
    return [int(mapping.get(c, 0)) for c in columns]


# ---------------------------------------------------------------------------
# Multi-target encoding
# ---------------------------------------------------------------------------

def multi_target_encode(train: pd.DataFrame, test: pd.DataFrame,
                        cols: list[str], y_tier: pd.Series, y_rate: pd.Series,
                        n_splits: int = 5, smoothing: float = 20.0,
                        seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode each high-cardinality categorical with THREE statistics per fold:
      - mean InterestRate (continuous risk)
      - P(tier=4) (subprime probability)
      - std InterestRate (within-group dispersion / uncertainty)
    This triples the signal per categorical versus a single mean encoding.
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    enc_tr = pd.DataFrame(index=train.index)
    enc_te = pd.DataFrame(index=test.index)
    gmean = float(y_rate.mean())
    gt4 = float((y_tier == 4).mean())
    gstd = float(y_rate.std())

    is_tier4 = (y_tier == 4).astype(int)

    for c in cols:
        # fold-wise encoding to avoid leakage
        mean_col = pd.Series(gmean, index=train.index, dtype=float)
        p4_col = pd.Series(gt4, index=train.index, dtype=float)
        std_col = pd.Series(gstd, index=train.index, dtype=float)
        for tri, vai in skf.split(train, y_tier):
            grp = train[c].iloc[tri]
            agg_r = y_rate.iloc[tri].groupby(grp).agg(["sum", "count", "std"])
            sm_mean = (agg_r["sum"] + smoothing * gmean) / (agg_r["count"] + smoothing)
            sm_std = agg_r["std"].fillna(gstd)
            agg_t = is_tier4.iloc[tri].groupby(grp).agg(["sum", "count"])
            sm_p4 = (agg_t["sum"] + smoothing * gt4) / (agg_t["count"] + smoothing)

            mean_col.iloc[vai] = train[c].iloc[vai].map(sm_mean).fillna(gmean).to_numpy()
            p4_col.iloc[vai] = train[c].iloc[vai].map(sm_p4).fillna(gt4).to_numpy()
            std_col.iloc[vai] = train[c].iloc[vai].map(sm_std).fillna(gstd).to_numpy()

        enc_tr[f"{c}_te_rate"] = mean_col
        enc_tr[f"{c}_te_p4"] = p4_col
        enc_tr[f"{c}_te_std"] = std_col

        # Full-train maps for test-time
        agg_r = y_rate.groupby(train[c]).agg(["sum", "count", "std"])
        sm_mean = (agg_r["sum"] + smoothing * gmean) / (agg_r["count"] + smoothing)
        sm_std = agg_r["std"].fillna(gstd)
        agg_t = is_tier4.groupby(train[c]).agg(["sum", "count"])
        sm_p4 = (agg_t["sum"] + smoothing * gt4) / (agg_t["count"] + smoothing)

        enc_te[f"{c}_te_rate"] = test[c].map(sm_mean).fillna(gmean).astype(float)
        enc_te[f"{c}_te_p4"] = test[c].map(sm_p4).fillna(gt4).astype(float)
        enc_te[f"{c}_te_std"] = test[c].map(sm_std).fillna(gstd).astype(float)

    return enc_tr, enc_te


# ---------------------------------------------------------------------------
# Tier-4 specialist (mixture-of-experts for Task B)
# ---------------------------------------------------------------------------

def train_tier4_mixture(X: pd.DataFrame, y_rate: pd.Series, y_tier: pd.Series,
                        X_test: pd.DataFrame, folds, rounds: int = 4000):
    """
    Mixture-of-experts rate regressor.

    Step 1: binary classifier P(tier = 4 | x).
    Step 2: "prime" regressor on tier 0–3 rows only (tight distribution).
    Step 3: "subprime" regressor on tier 4 rows only (long-tail).
    Final prediction: P(4)·subprime + (1 − P(4))·prime.

    Returns (oof, test_pred) in the same shape as any other Task B model, so
    it can plug straight into the existing ensemble.
    """
    is_4 = (y_tier == 4).astype(int)
    oof_rate = np.zeros(len(X))
    test_rate = np.zeros(len(X_test))

    for fi, (tri, vai) in enumerate(folds):
        # --- Stage 1: binary classifier ---
        cls_params = dict(
            objective="binary", metric="auc", learning_rate=0.03,
            num_leaves=127, min_child_samples=15, feature_fraction=0.75,
            bagging_fraction=0.80, bagging_freq=5, lambda_l2=1.0,
            verbose=-1, seed=SEED,
        )
        d_tr = lgb.Dataset(X.iloc[tri], label=is_4.iloc[tri], free_raw_data=False)
        d_va = lgb.Dataset(X.iloc[vai], label=is_4.iloc[vai], free_raw_data=False)
        cls = lgb.train(cls_params, d_tr, rounds, valid_sets=[d_va],
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        p4_va = cls.predict(X.iloc[vai], num_iteration=cls.best_iteration)
        p4_te = cls.predict(X_test, num_iteration=cls.best_iteration)

        # --- Stage 2: prime regressor (tiers 0-3 rows of train fold) ---
        prime_idx = np.array([i for i in tri if y_tier.iloc[i] != 4])
        sub_idx = np.array([i for i in tri if y_tier.iloc[i] == 4])
        reg_params = dict(
            objective="regression_l1", metric="rmse", learning_rate=0.02,
            num_leaves=127, min_child_samples=15, feature_fraction=0.75,
            bagging_fraction=0.80, bagging_freq=5, lambda_l2=1.0,
            verbose=-1, seed=SEED,
        )
        prime_tr = lgb.Dataset(X.iloc[prime_idx], label=y_rate.iloc[prime_idx],
                               free_raw_data=False)
        # Fresh Dataset per model — LightGBM frees raw data after use so we
        # build a new validation Dataset for each consumer.
        prime_va = lgb.Dataset(X.iloc[vai], label=y_rate.iloc[vai],
                               free_raw_data=False)
        prime = lgb.train(reg_params, prime_tr, rounds, valid_sets=[prime_va],
                          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        prime_va_pred = prime.predict(X.iloc[vai], num_iteration=prime.best_iteration)
        prime_te_pred = prime.predict(X_test, num_iteration=prime.best_iteration)

        # --- Stage 3: subprime regressor (tier 4 only) ---
        sub_tr = lgb.Dataset(X.iloc[sub_idx], label=y_rate.iloc[sub_idx],
                             free_raw_data=False)
        sub_va = lgb.Dataset(X.iloc[vai], label=y_rate.iloc[vai],
                             free_raw_data=False)  # fresh
        sub = lgb.train(reg_params, sub_tr, rounds, valid_sets=[sub_va],
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        sub_va_pred = sub.predict(X.iloc[vai], num_iteration=sub.best_iteration)
        sub_te_pred = sub.predict(X_test, num_iteration=sub.best_iteration)

        # --- Combine ---
        oof_rate[vai] = p4_va * sub_va_pred + (1 - p4_va) * prime_va_pred
        test_rate += (p4_te * sub_te_pred + (1 - p4_te) * prime_te_pred) / len(folds)

    return oof_rate, test_rate


# ---------------------------------------------------------------------------
# Two-stage Task A: binary is_tier4, then 4-class on non-tier-4
# ---------------------------------------------------------------------------

def train_two_stage_tier(X: pd.DataFrame, y_tier: pd.Series,
                         X_test: pd.DataFrame, folds, rounds: int = 6000):
    """
    Returns (oof_probs, test_probs) with shape (N, 5) — compatible with the
    other Task A classifiers. Internally uses the product P(k|x) = P(tier4|x) · 1[k=4]
                         + P(non-tier4|x) · P(k|x, non-tier4).
    The 4-class problem becomes much easier once tier 4 is filtered out.
    """
    is_4 = (y_tier == 4).astype(int)
    oof = np.zeros((len(X), N_CLASSES))
    test_p = np.zeros((len(X_test), N_CLASSES))

    for fi, (tri, vai) in enumerate(folds):
        # Stage 1: binary
        bin_params = dict(
            objective="binary", metric="auc", learning_rate=0.03,
            num_leaves=127, min_child_samples=15, feature_fraction=0.75,
            bagging_fraction=0.80, bagging_freq=5, lambda_l2=1.0,
            verbose=-1, seed=SEED,
        )
        d_tr = lgb.Dataset(X.iloc[tri], label=is_4.iloc[tri])
        d_va = lgb.Dataset(X.iloc[vai], label=is_4.iloc[vai])
        bin_m = lgb.train(bin_params, d_tr, rounds, valid_sets=[d_va],
                          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        p4_va = bin_m.predict(X.iloc[vai], num_iteration=bin_m.best_iteration)
        p4_te = bin_m.predict(X_test, num_iteration=bin_m.best_iteration)

        # Stage 2: 4-class on non-tier-4 rows
        prime_idx = np.array([i for i in tri if y_tier.iloc[i] != 4])
        y_prime = y_tier.iloc[prime_idx].to_numpy()  # values ∈ {0,1,2,3}
        multi_params = dict(
            objective="multiclass", num_class=4, metric="multi_logloss",
            learning_rate=0.02, num_leaves=127, min_child_samples=15,
            feature_fraction=0.75, bagging_fraction=0.80, bagging_freq=5,
            lambda_l2=1.0, verbose=-1, seed=SEED,
        )
        # Use the prime_idx slice of vai? No — validate on full vai to get
        # consistent OOF length. We'll give the 4-class model all vai rows
        # and let the binary gate handle tier-4s. Its output for tier-4 rows
        # is ignored after the gate.
        m4_tr = lgb.Dataset(X.iloc[prime_idx], label=y_prime)
        # Need a valid eval set; use prime rows inside vai if any, else all vai.
        val_prime = np.array([i for i in vai if y_tier.iloc[i] != 4])
        if len(val_prime) >= 50:
            m4_va = lgb.Dataset(X.iloc[val_prime], label=y_tier.iloc[val_prime])
        else:
            m4_va = lgb.Dataset(X.iloc[vai], label=y_tier.iloc[vai].clip(0, 3))
        m4 = lgb.train(multi_params, m4_tr, rounds, valid_sets=[m4_va],
                       callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        p03_va = m4.predict(X.iloc[vai], num_iteration=m4.best_iteration)  # (N_va, 4)
        p03_te = m4.predict(X_test, num_iteration=m4.best_iteration)

        # Combine: P(k<4|x)·P(k|x, non4), and P(tier4|x) for k=4
        oof[vai, 0:4] = (1 - p4_va)[:, None] * p03_va
        oof[vai, 4] = p4_va
        test_p[:, 0:4] += ((1 - p4_te)[:, None] * p03_te) / len(folds)
        test_p[:, 4] += p4_te / len(folds)

    return oof, test_p


# ---------------------------------------------------------------------------
# DART boosting variants (diversity)
# ---------------------------------------------------------------------------

def train_dart_cls(X, y, X_test, folds, rounds: int = 3500):
    """LightGBM DART variant for classification — different regularisation → diversity."""
    p = dict(
        boosting_type="dart", objective="multiclass", num_class=N_CLASSES,
        metric="multi_logloss", learning_rate=0.05, num_leaves=63,
        drop_rate=0.1, skip_drop=0.5, max_drop=50,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=1.0, verbose=-1, seed=SEED,
    )
    oof = np.zeros((len(X), N_CLASSES))
    tp = np.zeros((len(X_test), N_CLASSES))
    for tri, vai in folds:
        # DART doesn't support early stopping reliably → fixed rounds
        m = lgb.train(p, lgb.Dataset(X.iloc[tri], y.iloc[tri]), rounds,
                      valid_sets=[lgb.Dataset(X.iloc[vai], y.iloc[vai])],
                      callbacks=[lgb.log_evaluation(0)])
        oof[vai] = m.predict(X.iloc[vai])
        tp += m.predict(X_test) / len(folds)
    return oof, tp


def train_dart_reg(X, y, X_test, folds, rounds: int = 3500):
    p = dict(
        boosting_type="dart", objective="regression_l1", metric="rmse",
        learning_rate=0.05, num_leaves=63, drop_rate=0.1, skip_drop=0.5,
        max_drop=50, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=5, lambda_l2=1.0, verbose=-1, seed=SEED,
    )
    oof = np.zeros(len(X))
    tp = np.zeros(len(X_test))
    for tri, vai in folds:
        m = lgb.train(p, lgb.Dataset(X.iloc[tri], y.iloc[tri]), rounds,
                      valid_sets=[lgb.Dataset(X.iloc[vai], y.iloc[vai])],
                      callbacks=[lgb.log_evaluation(0)])
        oof[vai] = m.predict(X.iloc[vai])
        tp += m.predict(X_test) / len(folds)
    return oof, tp


# ---------------------------------------------------------------------------
# Quantile regression features (stage-2 input)
# ---------------------------------------------------------------------------

def train_quantile_reg(X, y, X_test, folds, quantiles=(0.1, 0.5, 0.9),
                       rounds: int = 3000):
    """
    Train one LightGBM regressor per quantile. Returns dict quantile → (oof, test).
    The spread q90 - q10 is a useful uncertainty feature for stage-2 blending —
    high spread indicates the row is in the subprime tail.
    """
    out = {}
    for q in quantiles:
        p = dict(
            objective="quantile", alpha=q, metric="quantile",
            learning_rate=0.03, num_leaves=127, min_child_samples=15,
            feature_fraction=0.75, bagging_fraction=0.80, bagging_freq=5,
            lambda_l2=1.0, verbose=-1, seed=SEED,
        )
        oof = np.zeros(len(X))
        tp = np.zeros(len(X_test))
        for tri, vai in folds:
            m = lgb.train(p, lgb.Dataset(X.iloc[tri], y.iloc[tri]), rounds,
                          valid_sets=[lgb.Dataset(X.iloc[vai], y.iloc[vai])],
                          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
            oof[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration)
            tp += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
        out[q] = (oof, tp)
    return out


# ---------------------------------------------------------------------------
# Ridge / Lasso meta-blender (unconstrained)
# ---------------------------------------------------------------------------

def ridge_blend_regression(oof_dict: dict, y: np.ndarray,
                           test_dict: dict | None = None,
                           alpha: float = 1.0):
    """
    Fit a Ridge regressor on stacked OOF predictions to produce final scalar
    predictions. Unlike convex blending, Ridge can assign negative/large
    coefficients, which often buys another +0.002-0.008 R² on top of the
    convex blend when the errors are correlated in specific ways.
    """
    names = list(oof_dict.keys())
    P = np.column_stack([oof_dict[n] for n in names])
    model = Ridge(alpha=alpha, positive=False, fit_intercept=True,
                  random_state=SEED)
    model.fit(P, y)
    blended = model.predict(P)
    test_blended = None
    if test_dict is not None:
        Pt = np.column_stack([test_dict[n] for n in names])
        test_blended = model.predict(Pt)
    return dict(zip(names, model.coef_)), float(model.intercept_), blended, test_blended


def ridge_blend_classification(oof_dict: dict, y: np.ndarray,
                               test_dict: dict | None = None,
                               alpha: float = 1.0):
    """
    Multi-class meta-blend: one Ridge per class predicting whether the
    true label is that class. Argmax of the K Ridge outputs gives the class.
    Strictly more expressive than convex averaging of probabilities.
    """
    names = list(oof_dict.keys())
    # Stack per-class probability columns across models → feature matrix (N, K·M)
    parts = []
    for n in names:
        parts.append(oof_dict[n])
    P = np.hstack(parts)  # (N, K*M)
    # One-hot target
    Y = np.eye(N_CLASSES)[y]
    model = Ridge(alpha=alpha, fit_intercept=True, random_state=SEED)
    model.fit(P, Y)
    blended = model.predict(P)  # (N, K)
    test_blended = None
    if test_dict is not None:
        parts_t = [test_dict[n] for n in names]
        Pt = np.hstack(parts_t)
        test_blended = model.predict(Pt)
    return model, blended, test_blended
