"""
features.py
Financially-grounded feature engineering tailored to the CreditSense schema.

Each function is defensive — it only builds a derived feature when its source
columns exist in the frame. Each feature carries a short comment with its
risk-modelling intuition; that commentary feeds directly into Section 2 of
the report.

The real dataset already provides LoanToIncomeRatio / MonthlyPaymentEstimate /
PaymentToIncomeRatio / TotalAssets, so we don't duplicate them — instead we
build complementary signals around them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _has(df: pd.DataFrame, *cols: str) -> bool:
    return all(c in df.columns for c in cols)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Division that treats 0 / 0 as 0 and x / 0 as NaN → filled with 0."""
    out = a.astype(float) / b.replace(0, np.nan).astype(float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ---------------------------------------------------------------------------
# 1. Leverage / balance-sheet features
# ---------------------------------------------------------------------------

def add_balance_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Net worth, liability stack, and liquidity ratios. In lending, a borrower
    with strong savings relative to the loan size has a cushion that lowers
    expected default — these ratios formalise that intuition.
    """
    out = df.copy()

    # Comprehensive liability stack
    liab_cols = [c for c in ["MortgageOutstandingBalance",
                             "StudentLoanOutstandingBalance",
                             "AutoLoanOutstandingBalance"] if c in out.columns]
    if liab_cols:
        out["feat_TotalLiabilities"] = out[liab_cols].fillna(0).sum(axis=1)

    # Net worth = assets - liabilities (TotalAssets already provided by dataset).
    if _has(out, "TotalAssets") and "feat_TotalLiabilities" in out.columns:
        out["feat_NetWorth"] = out["TotalAssets"] - out["feat_TotalLiabilities"]
        out["feat_DebtToAssets"] = _safe_div(out["feat_TotalLiabilities"],
                                             out["TotalAssets"])

    # Liquid vs illiquid wealth (cash is more protective than equity in a crisis).
    if _has(out, "SavingsBalance", "CheckingBalance"):
        out["feat_LiquidCash"] = out["SavingsBalance"].fillna(0) + \
                                 out["CheckingBalance"].fillna(0)
    if _has(out, "feat_LiquidCash", "RequestedLoanAmount"):
        out["feat_LiquidToLoan"] = _safe_div(out["feat_LiquidCash"],
                                             out["RequestedLoanAmount"])
    if _has(out, "feat_LiquidCash", "TotalMonthlyIncome"):
        # Months of expenses covered by cash (wealthy-but-illiquid vs wealthy-and-liquid).
        out["feat_CashRunwayMonths"] = _safe_div(out["feat_LiquidCash"],
                                                 out["TotalMonthlyIncome"])

    # Mortgage LTV — residual leverage on primary residence.
    if _has(out, "MortgageOutstandingBalance", "PropertyValue"):
        out["feat_MortgageLTV"] = _safe_div(out["MortgageOutstandingBalance"],
                                            out["PropertyValue"])

    # Investment share of wealth — diversified wealth is lower-risk.
    if _has(out, "InvestmentPortfolioValue", "TotalAssets"):
        out["feat_InvestShare"] = _safe_div(out["InvestmentPortfolioValue"],
                                            out["TotalAssets"])

    # Post-disbursement liquidity — what's left in the bank after the loan?
    if _has(out, "SavingsBalance", "RequestedLoanAmount"):
        out["feat_PostLoanSavings"] = (out["SavingsBalance"].fillna(0) -
                                       out["RequestedLoanAmount"])

    return out


# ---------------------------------------------------------------------------
# 2. Credit behaviour (delinquency + derogatory + inquiries)
# ---------------------------------------------------------------------------

def add_credit_behaviour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Payment history features. The 1/3/9 weighting on 30/60/90-day late
    payments follows FICO-style severity — a 90-day late is roughly an
    order of magnitude more predictive than a 30-day late.

    Charge-offs, collections, public records and bankruptcies are **derogatory
    marks**; in real underwriting each one shifts you down one risk tier.
    """
    out = df.copy()

    late30 = out.get("NumberOfLatePayments30Days", pd.Series(0, index=out.index))
    late60 = out.get("NumberOfLatePayments60Days", pd.Series(0, index=out.index))
    late90 = out.get("NumberOfLatePayments90Days", pd.Series(0, index=out.index))
    out["feat_DelinquencyScore"] = (1 * late30.fillna(0)
                                    + 3 * late60.fillna(0)
                                    + 9 * late90.fillna(0))
    out["feat_AnyDelinquency"] = (
        (late30.fillna(0) + late60.fillna(0) + late90.fillna(0)) > 0
    ).astype(np.int8)
    out["feat_SevereDelinquency"] = (late90.fillna(0) > 0).astype(np.int8)

    # Derogatory stack: a single weighted score capturing default-event history.
    deroc_cols = {"NumberOfChargeOffs": 5, "NumberOfCollections": 3,
                  "NumberOfPublicRecords": 4, "NumberOfBankruptcies": 10}
    score = pd.Series(0.0, index=out.index)
    any_deroc = pd.Series(0, index=out.index)
    for c, w in deroc_cols.items():
        if c in out.columns:
            v = out[c].fillna(0)
            score = score + w * v
            any_deroc = any_deroc + (v > 0).astype(int)
    out["feat_DerogatoryScore"] = score
    out["feat_AnyDerogatory"] = (any_deroc > 0).astype(np.int8)

    # Combined "bad events" composite — often the single strongest tier predictor.
    out["feat_BadEventsTotal"] = out["feat_DelinquencyScore"] + out["feat_DerogatoryScore"]

    # Credit inquiries — recent hard inquiries indicate active credit-seeking,
    # a short-term risk spike even for otherwise clean borrowers.
    hi12 = out.get("NumberOfHardInquiries12Mo",
                   pd.Series(0, index=out.index)).fillna(0)
    hi24 = out.get("NumberOfHardInquiries24Mo",
                   pd.Series(0, index=out.index)).fillna(0)
    out["feat_RecentInquiryShare"] = _safe_div(hi12, hi24)
    out["feat_HighInquiryActivity"] = (hi12 > 3).astype(np.int8)

    # Revolving utilisation — >90% is the single strongest near-term default signal.
    if "RevolvingUtilizationRate" in out.columns:
        util = out["RevolvingUtilizationRate"].fillna(0)
        out["feat_HighUtilization"] = (util > 0.9).astype(np.int8)
        out["feat_MaxedOut"] = (util >= 1.0).astype(np.int8)
        out["feat_UtilBucket"] = pd.cut(
            util, bins=[-0.01, 0.1, 0.3, 0.6, 0.9, np.inf], labels=False
        ).astype(np.int8)

    # Account quality ratio — fraction of accounts in good standing.
    if _has(out, "NumberOfSatisfactoryAccounts", "NumberOfOpenAccounts"):
        out["feat_SatisfactoryRatio"] = _safe_div(
            out["NumberOfSatisfactoryAccounts"], out["NumberOfOpenAccounts"])

    # Credit utilisation of limit (dollars used vs dollars available).
    if _has(out, "TotalCreditLimit", "RevolvingUtilizationRate"):
        out["feat_CreditUsed"] = (out["TotalCreditLimit"].fillna(0) *
                                  out["RevolvingUtilizationRate"].fillna(0))

    # Credit mix — more accounts + longer history = thicker file = lower risk.
    if _has(out, "NumberOfOpenAccounts", "NumberOfCreditCards"):
        out["feat_InstallmentAccounts"] = (out["NumberOfOpenAccounts"].fillna(0) -
                                           out["NumberOfCreditCards"].fillna(0))

    return out


# ---------------------------------------------------------------------------
# 3. Credit history maturity
# ---------------------------------------------------------------------------

def add_history_maturity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Length of credit history and age of oldest account. "Thin file" borrowers
    (short history, few accounts) are systematically riskier even at matched
    income and DTI — this effect is well documented in credit-scoring research.
    """
    out = df.copy()

    if "CreditHistoryLengthMonths" in out.columns:
        chl = out["CreditHistoryLengthMonths"].fillna(0)
        out["feat_ThinFile"] = (chl < 24).astype(np.int8)     # < 2 years
        out["feat_ThickFile"] = (chl > 120).astype(np.int8)   # > 10 years

    if _has(out, "OldestAccountAgeMonths", "AverageAccountAgeMonths"):
        # If average age << oldest age, borrower opened many new accounts
        # recently (thin recent behaviour on top of an old anchor account).
        out["feat_NewAccountSpree"] = (
            out["OldestAccountAgeMonths"].fillna(0) -
            out["AverageAccountAgeMonths"].fillna(0)
        )

    if _has(out, "Age", "CreditHistoryLengthMonths"):
        # Fraction of adult life with credit — late-starters look like young borrowers.
        out["feat_CreditAgeRatio"] = _safe_div(
            out["CreditHistoryLengthMonths"],
            (out["Age"].fillna(25) - 18).clip(lower=1) * 12,
        )

    return out


# ---------------------------------------------------------------------------
# 4. Demographics + employment stability
# ---------------------------------------------------------------------------

def add_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Employment tenure, residency stability, dependents. These capture
    socioeconomic stability that is weakly reflected in pure credit metrics.
    """
    out = df.copy()

    if "Age" in out.columns:
        out["feat_AgeBucket"] = pd.cut(
            out["Age"].fillna(out["Age"].median()),
            bins=[-1, 25, 35, 45, 55, 65, 200], labels=False,
        ).astype(np.int8)

    if _has(out, "EmploymentLengthYears", "Age"):
        out["feat_EmpStability"] = _safe_div(
            out["EmploymentLengthYears"].fillna(0),
            (out["Age"] - 18).clip(lower=1),
        )

    if _has(out, "AnnualIncome", "EmploymentLengthYears"):
        out["feat_IncomePerEmpYear"] = _safe_div(
            out["AnnualIncome"], out["EmploymentLengthYears"].fillna(0) + 1)

    if _has(out, "YearsAtCurrentEmployer", "EmploymentLengthYears"):
        # Same employer vs job hopper — job hoppers at same cumulative tenure
        # are empirically a bit riskier.
        out["feat_SameEmployerShare"] = _safe_div(
            out["YearsAtCurrentEmployer"].fillna(0),
            out["EmploymentLengthYears"].fillna(0) + 0.5,
        ).clip(0, 1)

    if "ResidencyYears" in out.columns:
        out["feat_StableResident"] = (out["ResidencyYears"].fillna(0) > 3
                                      ).astype(np.int8)

    if "HomeOwnership" in out.columns:
        owns = out["HomeOwnership"].astype(str).str.upper().str.contains(
            "OWN|MORTGAGE", regex=True, na=False)
        out["feat_IsHomeOwner"] = owns.astype(np.int8)

    if "NumberOfDependents" in out.columns:
        out["feat_HasDependents"] = (out["NumberOfDependents"].fillna(0) > 0
                                     ).astype(np.int8)
    if _has(out, "NumberOfDependentsUnder18", "NumberOfDependents"):
        out["feat_KidsShare"] = _safe_div(
            out["NumberOfDependentsUnder18"], out["NumberOfDependents"])

    if "IncomeVerified" in out.columns:
        # Verified income is a strong trust signal — unverified applications
        # are systematically riskier at matched stated income.
        out["feat_IncomeVerified"] = out["IncomeVerified"].fillna(0).astype(np.int8)

    if _has(out, "SecondaryMonthlyIncome", "MonthlyGrossIncome"):
        out["feat_HasSecondaryIncome"] = (
            out["SecondaryMonthlyIncome"].fillna(0) > 0).astype(np.int8)
        out["feat_SecondaryIncomeShare"] = _safe_div(
            out["SecondaryMonthlyIncome"].fillna(0),
            out["MonthlyGrossIncome"] + out["SecondaryMonthlyIncome"].fillna(0))

    return out


# ---------------------------------------------------------------------------
# 5. Loan request
# ---------------------------------------------------------------------------

def add_loan_request(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features derived from the requested-loan block. MonthlyPaymentEstimate and
    PaymentToIncomeRatio are already provided, so we add complementary signals.
    """
    out = df.copy()

    if _has(out, "RequestedLoanAmount", "RequestedTermMonths"):
        out["feat_MonthlyPrincipal"] = _safe_div(
            out["RequestedLoanAmount"], out["RequestedTermMonths"])

    if "RequestedTermMonths" in out.columns:
        out["feat_LongTerm"] = (out["RequestedTermMonths"] >= 60).astype(np.int8)
        out["feat_ShortTerm"] = (out["RequestedTermMonths"] <= 24).astype(np.int8)

    if _has(out, "RequestedLoanAmount", "TotalAssets"):
        # Loan size relative to existing assets — unusual values flag stretched applicants.
        out["feat_LoanToAssets"] = _safe_div(out["RequestedLoanAmount"],
                                             out["TotalAssets"])

    if _has(out, "MonthlyPaymentEstimate", "feat_LiquidCash"):
        # Months of payment covered by liquid cash — a "reserve" measure.
        out["feat_PaymentReserveMonths"] = _safe_div(out["feat_LiquidCash"],
                                                     out["MonthlyPaymentEstimate"])

    if "CollateralType" in out.columns:
        out["feat_HasCollateral"] = (
            out["CollateralType"].astype(str).str.upper().str.strip()
            .isin(["NONE", "NAN", "NA", ""])  # None-like values
            .eq(False)
        ).astype(np.int8)

    if "HasCoApplicant" in out.columns:
        out["feat_HasCoApplicant"] = out["HasCoApplicant"].fillna(0).astype(np.int8)

    if "PreviousLoanWithBank" in out.columns:
        # Repeat customers have observable repayment history → lower risk prior.
        out["feat_RepeatCustomer"] = out["PreviousLoanWithBank"].fillna(0).astype(np.int8)

    return out


# ---------------------------------------------------------------------------
# 6. Log-transforms and interactions
# ---------------------------------------------------------------------------

def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Money-denominated variables are right-skewed log-normal → log1p helps."""
    out = df.copy()
    money_like = [c for c in out.columns
                  if not c.startswith("feat_log_")
                  and any(k in c.lower() for k in
                          ("income", "amount", "balance", "value", "asset"))
                  and pd.api.types.is_numeric_dtype(out[c])]
    for c in money_like:
        out[f"feat_log_{c}"] = np.log1p(out[c].fillna(0).clip(lower=0))
    return out


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Non-linear interactions known to matter in consumer credit risk."""
    out = df.copy()

    if _has(out, "DebtToIncomeRatio", "RevolvingUtilizationRate"):
        # High DTI × high utilisation compounds default risk.
        out["feat_DTI_x_Util"] = (out["DebtToIncomeRatio"].fillna(0)
                                  * out["RevolvingUtilizationRate"].fillna(0))

    if _has(out, "LoanToIncomeRatio", "RequestedTermMonths"):
        # Long term × high LTI = payment fragility under income shock.
        out["feat_LTI_x_Term"] = (out["LoanToIncomeRatio"].fillna(0)
                                  * out["RequestedTermMonths"].fillna(0))

    if _has(out, "Age", "feat_DelinquencyScore"):
        out["feat_DelinquencyPerYear"] = _safe_div(
            out["feat_DelinquencyScore"], (out["Age"] - 18).clip(lower=1))

    if _has(out, "PaymentToIncomeRatio", "feat_BadEventsTotal"):
        # High PTI × bad history is the textbook "expected default" combo.
        out["feat_PTI_x_BadEvents"] = (out["PaymentToIncomeRatio"].fillna(0)
                                       * out["feat_BadEventsTotal"])

    if _has(out, "CreditHistoryLengthMonths", "feat_DerogatoryScore"):
        # Recent derogatory on a thin file is worse than on a thick one.
        out["feat_DerogPerHistoryYear"] = _safe_div(
            out["feat_DerogatoryScore"],
            out["CreditHistoryLengthMonths"].fillna(0) / 12 + 1)

    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline; order matters (interactions reference earlier feats)."""
    out = df
    out = add_balance_sheet(out)
    out = add_credit_behaviour(out)
    out = add_history_maturity(out)
    out = add_demographics(out)
    out = add_loan_request(out)
    out = add_log_transforms(out)
    out = add_interactions(out)
    out = out.replace([np.inf, -np.inf], np.nan)
    # Only fill numeric NaNs; preserve categorical NaN for the encoder layer.
    num_cols = out.select_dtypes(include=np.number).columns
    out[num_cols] = out[num_cols].fillna(0.0)
    return out
