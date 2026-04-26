"""
utils.py
Shared utilities: deterministic seeding, cross-validation splits, scoring,
and a K-fold target encoder that avoids leakage.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OOF_DIR = OUT_DIR / "oof"
MODELS_DIR = OUT_DIR / "models"

TARGET_A = "RiskTier"
TARGET_B = "InterestRate"
ID_COL = "Id"
RATE_MIN, RATE_MAX = 4.99, 35.99


def set_seed(seed: int = SEED) -> None:
    """Make every stochastic component reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data():
    """Load train + test CSVs. Expects files in ml_final/data/.
    The Kaggle CSVs don't carry an explicit Id column — sample_submission.csv
    uses Id = row index in credit_test.csv. We synthesise it on load so
    downstream code can treat it uniformly.
    """
    train_path = DATA_DIR / "credit_train.csv"
    test_path = DATA_DIR / "credit_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing data files. Expected {train_path} and {test_path}. "
            f"See README.txt for Kaggle download instructions."
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if ID_COL not in train.columns:
        train.insert(0, ID_COL, np.arange(len(train)))
    if ID_COL not in test.columns:
        test.insert(0, ID_COL, np.arange(len(test)))
    return train, test


def get_folds(y_stratify: pd.Series, n_splits: int = N_FOLDS, seed: int = SEED):
    """
    Return a list of (train_idx, val_idx) tuples.
    Stratifies on RiskTier so Task A class balance is preserved; the same splits
    are reused for Task B to keep OOF predictions aligned across tasks.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.arange(len(y_stratify)), y_stratify))


def combined_score(y_tier_true, y_tier_pred, y_rate_true, y_rate_pred) -> float:
    """Kaggle leaderboard: 0.5 * Accuracy + 0.5 * R^2."""
    acc = accuracy_score(y_tier_true, y_tier_pred)
    r2 = r2_score(y_rate_true, y_rate_pred)
    return 0.5 * acc + 0.5 * r2


def clip_rate(preds: np.ndarray) -> np.ndarray:
    """Clip predicted interest rate to the legal APR range and round to 2dp."""
    return np.round(np.clip(preds, RATE_MIN, RATE_MAX), 2)


class KFoldTargetEncoder:
    """
    Leakage-safe target encoding. During .fit_transform, each row's encoding is
    computed only from rows in OTHER folds. At .transform (test) time we use the
    full-train mean. Smoothing regularises rare categories toward the global mean.
    """

    def __init__(self, cols, n_splits: int = N_FOLDS, smoothing: float = 10.0,
                 seed: int = SEED):
        self.cols = list(cols)
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.seed = seed
        self.global_mean_: float | None = None
        self.full_mapping_: dict[str, pd.Series] = {}

    def _smoothed_mean(self, group_sum, group_count, global_mean, smoothing):
        return (group_sum + smoothing * global_mean) / (group_count + smoothing)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, stratify: pd.Series):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.seed)
        encoded = pd.DataFrame(index=X.index)
        self.global_mean_ = float(y.mean())

        for col in self.cols:
            out = pd.Series(self.global_mean_, index=X.index, dtype=float)
            for tr_idx, va_idx in skf.split(X, stratify):
                tr_y = y.iloc[tr_idx]
                tr_x = X[col].iloc[tr_idx]
                agg = tr_y.groupby(tr_x).agg(["sum", "count"])
                smoothed = self._smoothed_mean(agg["sum"], agg["count"],
                                               self.global_mean_, self.smoothing)
                out.iloc[va_idx] = X[col].iloc[va_idx].map(smoothed).fillna(
                    self.global_mean_).to_numpy()
            encoded[f"{col}_te"] = out

            # Full-train mapping for test-time transform
            agg_all = y.groupby(X[col]).agg(["sum", "count"])
            self.full_mapping_[col] = self._smoothed_mean(
                agg_all["sum"], agg_all["count"], self.global_mean_, self.smoothing
            )
        return encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.DataFrame(index=X.index)
        for col in self.cols:
            encoded[f"{col}_te"] = X[col].map(self.full_mapping_[col]).fillna(
                self.global_mean_).astype(float)
        return encoded


def print_header(msg: str) -> None:
    """Small helper so console output is scannable during long runs."""
    line = "=" * max(60, len(msg) + 4)
    print(f"\n{line}\n  {msg}\n{line}")
