# CreditSense — Written Report Outline (5 pages, PDF)

**Team:** Slavs — Makar Ulesov, Ivan Kanev, Delyan Hristov
**Course:** AI1215 Introduction to Machine Learning, Spring 2026
**Kaggle usernames:** TBD

---

## Cover page
- Team name, member names, Kaggle username(s), final Kaggle combined score, date.

---

## Section 1 — Data Exploration & Preprocessing (30%)

**Key findings (fill from `code/01_eda.ipynb` after training):**
- RiskTier is near-perfectly balanced (6.7k–7.3k per class), so we use unweighted multiclass loss and do not need SMOTE.
- InterestRate is right-skewed with a floor at 4.99% — a large fraction of borrowers receive the best rate, meaning the model must distinguish "prime" borrowers from everyone else.
- Monotone scatter of `InterestRate` vs `RiskTier` confirms the two tasks are tightly coupled → motivates cross-task stacking (Section 3).
- Structural missingness is confirmed:
  - `PropertyValue` / `MortgageOutstandingBalance` missing ⇔ renter.
  - `StudentLoanOutstandingBalance` missing ⇔ older applicants without student debt.
  - `CollateralType` / `CollateralValue` missing ⇔ unsecured loan.
  - We keep this as *signal* via `*_was_missing` binary indicators rather than erase it with naive imputation.

**Preprocessing choices and why:**
| Step | Choice | Rationale |
|---|---|---|
| Missing indicators | add `{col}_was_missing` binary for every column with NaN | Structural missingness carries risk signal; indicators preserve it when the column itself is then imputed. |
| Numeric imputation | `STRUCTURAL_ZERO`→0, otherwise median | Zero is the semantically correct value for "renter's mortgage balance"; median only for truly unknown values. |
| Outliers | winsorise money-like columns at 99.5th percentile | Avoids tree splits being dominated by a handful of extreme earners. |
| Ordinal mapping | `EducationLevel` → integer rank | Preserves the ordinal relationship that one-hot would destroy. |
| Low-card nominal | one-hot (cardinality ≤ 8) | Safe when dimensions stay small. |
| High-card nominal (`State`, `JobCategory`, `LoanPurpose`) | 5-fold target encoding with smoothing 20 + integer codes | Target encoding uses the InterestRate signal; K-fold prevents leakage. We *also* keep integer codes so LightGBM can use them as categorical features natively. |

**Visualisation requirement (one of):** target joint scatter, missingness heatmap, log1p income distribution.

---

## Section 2 — Feature Engineering (35%)

We group 40+ engineered features into six blocks with clear financial intuition.

**Balance-sheet block:**
- `feat_NetWorth`, `feat_DebtToAssets`, `feat_LiquidCash`, `feat_LiquidToLoan`, `feat_CashRunwayMonths`, `feat_MortgageLTV`, `feat_InvestShare`, `feat_PostLoanSavings`.
- *Intuition:* underwriters care about leverage, liquidity, and buffer — not just income.

**Credit-behaviour block:**
- `feat_DelinquencyScore = 1·Late30 + 3·Late60 + 9·Late90` (FICO-style severity).
- `feat_DerogatoryScore` with weights (10·Bankruptcy + 5·ChargeOff + 4·PublicRecord + 3·Collection).
- `feat_BadEventsTotal` = delinquency + derogatory — single strongest RiskTier predictor.
- `feat_HighUtilization`, `feat_MaxedOut`, `feat_UtilBucket`, `feat_RecentInquiryShare`, `feat_HighInquiryActivity`, `feat_SatisfactoryRatio`, `feat_CreditUsed`.

**History-maturity block:**
- `feat_ThinFile` (< 24 months), `feat_ThickFile` (> 120 months), `feat_NewAccountSpree`, `feat_CreditAgeRatio`.
- *Intuition:* thin-file borrowers are systematically riskier at matched DTI — classic credit-scoring result.

**Demographics + stability block:**
- `feat_AgeBucket`, `feat_EmpStability`, `feat_IncomePerEmpYear`, `feat_SameEmployerShare`, `feat_StableResident`, `feat_IsHomeOwner`, `feat_HasDependents`, `feat_KidsShare`, `feat_IncomeVerified`, `feat_HasSecondaryIncome`.

**Loan-request block:**
- `feat_MonthlyPrincipal`, `feat_LongTerm`, `feat_ShortTerm`, `feat_LoanToAssets`, `feat_PaymentReserveMonths`, `feat_HasCollateral`, `feat_HasCoApplicant`, `feat_RepeatCustomer`.

**Transforms + interactions:**
- `feat_log_*` for every money-denominated column (tames right skew).
- Interactions: `feat_DTI_x_Util`, `feat_LTI_x_Term`, `feat_DelinquencyPerYear`, `feat_PTI_x_BadEvents`, `feat_DerogPerHistoryYear`.

**Ablation table (fill after full run completes):**
| Feature block added | Task A OOF Acc | Task B OOF R² | Combined |
|---|---|---|---|
| Raw columns + one-hot baseline | — | — | — |
| + Balance-sheet | — | — | — |
| + Credit behaviour | — | — | — |
| + History maturity | — | — | — |
| + Demographics + loan-request | — | — | — |
| + Log-transforms + interactions | — | — | — |

---

## Section 3 — Model Selection & Tuning (35%)

**Models compared:**
| Model | Task A Acc (OOF) | Task B R² (OOF) |
|---|---|---|
| Logistic / Linear Regression (sanity) | — | — |
| LightGBM (multiclass / regression) | — | — |
| XGBoost | — | — |
| CatBoost | — | — |
| LightGBM regression-on-tier (ordinal trick) | — | — |
| **Stage-2 LightGBM on augmented features** | — | — |
| **Final convex ensemble** | — | — |

**CV protocol:**
- 5-fold `StratifiedKFold` on `RiskTier`, seed 42.
- Same fold indices reused for Task B so OOF arrays align and we can measure the combined leaderboard score directly on OOF.
- Early stopping 150 rounds on held-out fold.

**Overfitting controls:**
- Early stopping per fold.
- K-fold target encoding (fold *i* encoded using folds ≠ *i*) prevents leakage from `InterestRate` into train features.
- `feature_fraction = 0.85`, `bagging_fraction = 0.85` on LightGBM for tree diversity.
- Winsorisation caps extreme predictor values.
- We monitor gap between OOF and Kaggle public score after each submission — flat gap = no leak.

**Cross-task stacking (novel contribution):**
We observed that RiskTier and InterestRate are nearly collinear in this dataset (monotone box-plot in EDA). We therefore:
1. Produce OOF predictions from 4 classifiers (LGB, XGB, CAT, ordinal-regression LGB) for Task A and 3 regressors for Task B.
2. Build augmented feature sets = original features + all OOF predictions.
3. Train a stage-2 LightGBM per task on the augmented set.
4. Blend stage-2 with stage-1 via convex weights optimised to maximise OOF `0.5·Acc + 0.5·R²`.

The key insight: feeding the OOF Task B rate into the Task A model (and vice-versa) gives each task access to the other's signal without target leakage — the OOF predictions only see each row from a held-out fold.

**Feature importance — Task A vs Task B:**
(Fill after training. Expectation: both tasks share top predictors (delinquency score, DTI, payment-to-income), but Task A ranks categorical signals higher (`HomeOwnership`, `LoanPurpose`) while Task B ranks continuous leverage ratios higher.)

**Final approach & results:**
- OOF combined: **[fill]**
- Kaggle public score: **[fill]**
- Gap OOF vs Kaggle: **[fill]** — a small positive gap indicates no leakage.

---

## Sources & references (short, cite what you used)

- PDF: AI1215 CreditSense Data Challenge assignment (Spring 2026).
- LightGBM paper: Ke et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017.
- XGBoost paper: Chen & Guestrin "XGBoost: A Scalable Tree Boosting System", KDD 2016.
- CatBoost: Prokhorenkova et al. "CatBoost: unbiased boosting with categorical features", NeurIPS 2018.
- Target encoding / K-fold leakage prevention: Micci-Barreca 2001 ("A Preprocessing Scheme for High-Cardinality Categorical Attributes").
- FICO severity weighting for late payments — standard industry convention.
