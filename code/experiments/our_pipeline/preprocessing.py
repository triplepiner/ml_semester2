"""
preprocessing.py
Missing-value indicators, contextual imputation, and categorical encoding.
Designed to be schema-defensive: it inspects dtypes rather than hard-coding
55-column names, so minor column-name drift in the Kaggle CSVs won't break it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import KFoldTargetEncoder, TARGET_A, TARGET_B, ID_COL


# Columns we know are ordinal from the PDF / common loan datasets. We only apply
# the mapping if the column exists AND its values fit (otherwise fall through to
# generic categorical handling).
ORDINAL_MAPS = {
    "EducationLevel": {
        "None": 0, "HighSchool": 1, "High School": 1, "Associate": 2,
        "Bachelor": 3, "Bachelors": 3, "Master": 4, "Masters": 4,
        "PhD": 5, "Doctorate": 5,
    },
    "EmploymentLengthYears": None,  # numeric in dataset; kept for reference
}

# Known numeric columns where missingness is structural and the "typical" value
# for a missing row is zero (not median). E.g., a renter has no PropertyValue.
STRUCTURAL_ZERO_CANDIDATES = [
    "PropertyValue", "MortgageOutstandingBalance",
    "StudentLoanOutstandingBalance", "AutoLoanOutstandingBalance",
    "InvestmentPortfolioValue", "VehicleValue",
    "CollateralValue", "SecondaryMonthlyIncome",
]


def split_features(df: pd.DataFrame):
    """Return (numeric_cols, categorical_cols) excluding targets + id."""
    exclude = {TARGET_A, TARGET_B, ID_COL}
    numeric = [c for c in df.columns
               if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns
                   if c not in exclude and c not in numeric]
    return numeric, categorical


def add_missing_indicators(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    For every column with ≥ 1 NaN, add a binary indicator. Missingness is a
    signal (PDF explicitly says so), so we keep it separately from imputation.
    """
    out = df.copy()
    for c in cols:
        if out[c].isna().any():
            out[f"{c}_was_missing"] = out[c].isna().astype(np.int8)
    return out


def impute_numeric(train: pd.DataFrame, test: pd.DataFrame,
                   numeric_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute numeric NaNs:
      - Structural-zero candidates (PropertyValue etc.): fill with 0.
      - Everything else: median fit on train, applied to both.
    """
    train = train.copy()
    test = test.copy()
    for c in numeric_cols:
        if c in STRUCTURAL_ZERO_CANDIDATES:
            fill = 0.0
        else:
            fill = train[c].median()
        train[c] = train[c].fillna(fill)
        test[c] = test[c].fillna(fill)
    return train, test


def apply_ordinal_maps(df: pd.DataFrame) -> pd.DataFrame:
    """Map ordinal categorical strings → integers when a mapping is known."""
    out = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if mapping is None or col not in out.columns:
            continue
        mapped = out[col].map(mapping)
        # Only adopt the mapping if it covers most values; otherwise skip and
        # let the column flow through as generic categorical.
        if mapped.notna().mean() > 0.8:
            out[col] = mapped.fillna(-1).astype(np.int16)
    return out


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame,
                        categorical_cols: list[str],
                        y_for_te: pd.Series,
                        stratify: pd.Series,
                        one_hot_threshold: int = 8,
                        te_smoothing: float = 20.0):
    """
    Encoding strategy:
      - Cardinality ≤ one_hot_threshold → one-hot.
      - Cardinality >  one_hot_threshold → leakage-safe K-fold target encoding
        (against InterestRate; works for both tasks because the two targets are
        nearly collinear).
    Returns aligned (train_enc, test_enc) DataFrames.
    """
    train = train.copy()
    test = test.copy()

    # NaNs in categoricals → literal "NA" token
    for c in categorical_cols:
        train[c] = train[c].fillna("NA").astype(str)
        test[c] = test[c].fillna("NA").astype(str)

    low_card = [c for c in categorical_cols if train[c].nunique() <= one_hot_threshold]
    high_card = [c for c in categorical_cols if c not in low_card]

    # One-hot — fit on the union of categories so train/test stay aligned
    oh_train = pd.get_dummies(train[low_card], prefix=low_card, dtype=np.int8) \
        if low_card else pd.DataFrame(index=train.index)
    oh_test = pd.get_dummies(test[low_card], prefix=low_card, dtype=np.int8) \
        if low_card else pd.DataFrame(index=test.index)
    oh_train, oh_test = oh_train.align(oh_test, join="outer", axis=1, fill_value=0)

    # Target encoding for high-cardinality columns
    if high_card:
        te = KFoldTargetEncoder(cols=high_card, smoothing=te_smoothing)
        te_train = te.fit_transform(train[high_card], y_for_te, stratify)
        te_test = te.transform(test[high_card])
    else:
        te_train = pd.DataFrame(index=train.index)
        te_test = pd.DataFrame(index=test.index)

    # Keep original high-card columns as categorical codes too — LightGBM and
    # CatBoost can exploit them natively alongside the TE signal.
    code_train = pd.DataFrame(index=train.index)
    code_test = pd.DataFrame(index=test.index)
    for c in high_card:
        combined = pd.concat([train[c], test[c]], axis=0)
        codes, _ = pd.factorize(combined, sort=True)
        code_train[f"{c}_code"] = codes[:len(train)].astype(np.int32)
        code_test[f"{c}_code"] = codes[len(train):].astype(np.int32)

    train_enc = pd.concat([train.drop(columns=categorical_cols),
                           oh_train, te_train, code_train], axis=1)
    test_enc = pd.concat([test.drop(columns=categorical_cols),
                          oh_test, te_test, code_test], axis=1)
    return train_enc, test_enc


def winsorize(train: pd.DataFrame, test: pd.DataFrame,
              cols: list[str], q: float = 0.995) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap extreme values at the qth percentile computed on train only."""
    train = train.copy()
    test = test.copy()
    for c in cols:
        if c not in train.columns:
            continue
        cap = train[c].quantile(q)
        train[c] = np.minimum(train[c], cap)
        test[c] = np.minimum(test[c], cap)
    return train, test


def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    """
    End-to-end preprocessing.
    Returns (train_proc, test_proc, y_tier, y_rate, ids_test).
    """
    y_tier = train[TARGET_A].astype(int).copy()
    y_rate = train[TARGET_B].astype(float).copy()
    ids_test = test[ID_COL].copy() if ID_COL in test.columns else pd.Series(
        range(len(test)), name=ID_COL)

    # Drop targets + id before feature work
    tr = train.drop(columns=[c for c in [TARGET_A, TARGET_B, ID_COL]
                             if c in train.columns]).copy()
    te = test.drop(columns=[c for c in [ID_COL] if c in test.columns]).copy()

    # Align columns in case test has a stray extra (defensive)
    common = [c for c in tr.columns if c in te.columns]
    tr, te = tr[common], te[common]

    numeric, categorical = split_features(tr.assign(**{TARGET_A: 0, TARGET_B: 0.0}))
    # split_features above expects targets to exist; strip them back out
    numeric = [c for c in numeric if c in tr.columns]
    categorical = [c for c in categorical if c in tr.columns]

    # Missing indicators BEFORE imputation so the signal is preserved
    all_feature_cols = numeric + categorical
    tr = add_missing_indicators(tr, all_feature_cols)
    te = add_missing_indicators(te, all_feature_cols)

    tr, te = impute_numeric(tr, te, numeric)

    # Winsorize money-like columns with heavy right tails
    money_like = [c for c in numeric if any(k in c.lower() for k in
                  ("income", "amount", "balance", "value", "loan"))]
    tr, te = winsorize(tr, te, money_like)

    tr = apply_ordinal_maps(tr)
    te = apply_ordinal_maps(te)

    # Post-ordinal, some columns may have become numeric; recompute split
    numeric, categorical = split_features(tr.assign(**{TARGET_A: 0, TARGET_B: 0.0}))
    numeric = [c for c in numeric if c in tr.columns]
    categorical = [c for c in categorical if c in tr.columns]

    tr_enc, te_enc = encode_categoricals(
        tr, te, categorical_cols=categorical,
        y_for_te=y_rate, stratify=y_tier,
    )

    # Final sanity: numeric dtypes, no NaN
    tr_enc = tr_enc.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    te_enc = te_enc.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    tr_enc, te_enc = tr_enc.align(te_enc, join="inner", axis=1)

    return tr_enc, te_enc, y_tier, y_rate, ids_test
