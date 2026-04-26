CreditSense - Loan Risk Assessment Challenge (AI1215, Spring 2026)
=====================================================================

Team name     : BULGARIA FOREVER
Team members  : Makar Ulesov, Ivan Kanev, Delyan Hristov
Kaggle user(s): makarulesov, Vizior, delyanhristov

Note on team name. The team was originally registered with the course
under the name "BULGARIA FOREVER" and is listed under that name on the
oral-presentation schedule. When we created the team on Kaggle we briefly
used the alternate name "Slavs"; we renamed the Kaggle team back to
"BULGARIA FOREVER" on 26 April 2026 to align with the course schedule.
All Kaggle submissions made under either name belong to the same three
team members listed above.

---------------------------------------------------------------------
1. Reproducing the submitted Kaggle CSV
---------------------------------------------------------------------

Prerequisites
  * Python 3.10, 3.11, 3.12 or 3.13 (tested on 3.13.5)
  * ~6 GB free RAM, ~500 MB disk for outputs
  * Jupyter (the canonical pipeline is a notebook)

Step 1 - download data
  Join the competition and download credit_train.csv + credit_test.csv:
      https://www.kaggle.com/t/3e62a127eb85418aa851a5ee258e7c04
  Place both CSVs inside  ml_final/data/  (create the folder if missing).

Step 2 - install dependencies
      python -m venv .venv
      source .venv/bin/activate            # Windows: .venv\Scripts\activate
      pip install -r requirements.txt

Step 3 - run the pipeline
      cd code/
      jupyter nbconvert --to notebook --execute boosting.ipynb \
              --output boosting.executed.ipynb

  The notebook is the canonical reproduction entry point. It runs end-to-end
  from the raw CSVs and writes  outputs/submission.csv  on completion.

  Expected runtime: 50-70 minutes on an 8-core CPU (Optuna tuning of six
  base learners per task is the bulk of the time).
  Expected combined CV score: ~0.86 (0.5 * Accuracy + 0.5 * R^2).
  All randomness is seeded with random_state=42 throughout.

Step 4 - submit to Kaggle
  Upload  outputs/submission.csv  to the competition page.

---------------------------------------------------------------------
2. File map
---------------------------------------------------------------------

  data/
    credit_train.csv         - downloaded from Kaggle
    credit_test.csv          - downloaded from Kaggle
  code/
    boosting.ipynb           - *** canonical pipeline (the one we shipped) ***
    eda.ipynb                - exploratory data analysis (feeds report Sec.1)
    boosting_PLAN.md         - design notes for the canonical pipeline
    experiments/             - parallel attempts and iteration history,
                               kept for traceability. NOT required for
                               the official reproduction. See its README.
  outputs/
    submission.csv           - Kaggle upload produced by boosting.ipynb
  report/
    CreditSense_Report.pdf   - 5-page written report
    CreditSense_Report.docx  - editable Word version
  README.txt
  requirements.txt

---------------------------------------------------------------------
3. Pipeline overview (boosting.ipynb)
---------------------------------------------------------------------

Cell 0    Imports (pandas, numpy, scikit-learn, xgboost, optuna).
Cell 1    Load credit_train.csv; split features / RiskTier / InterestRate.
Cell 2    Smart preprocessing:
            * Zero-fill 8 structural columns where NaN means "doesn't have
              one" (PropertyValue for renters, StudentLoan for older
              applicants, CollateralValue for unsecured loans, etc.).
            * Missing-flag only on the 2 columns where missingness shifts
              the target meaningfully (InvestmentPortfolioValue,
              RevolvingUtilizationRate). Per EDA — most other missingness
              flags add noise.
            * EducationLevel mapped to integer ordinal (preserves the
              monotonic RiskTier shift across levels).
            * Drop State, JobCategory, MaritalStatus (no signal in EDA).
            * Five engineered aggregates from underwriter heuristics:
              TotalLatePayments, DerogMarks, SatisfactoryAccountRatio,
              RiskSeverityScore (1*30d + 3*60d + 9*90d + 15*ChargeOffs +
              10*Collections + 30*Bankruptcies + 8*PublicRecords).
Cell 3    4-signal feature selection:
            * XGBoost gain importance on RiskTier
            * XGBoost gain importance on InterestRate
            * Mutual information vs RiskTier
            * Mutual information vs InterestRate
          Drop a feature only if it scores low on ALL four signals. Then
          correlation-prune at |rho| > 0.95. Result: 18 features (down
          from 55 raw + 5 engineered = 63).
Cell 4    Optuna TPE tuning per base learner (3-fold CV):
            xgb1 (25 trials), xgb2 (25), rf (15), et (15), hgb (20),
            mlp (20).
Cell 5    Task A stacked ensemble: 6 tuned base + LogisticRegression meta,
          5-fold CV. CV accuracy = 88.12% (+/- 0.87%).
Cell 6    OOF RiskTier features: 5-fold cross_val_predict produces leak-
          free probability matrix; expected_tier = sum(p_i * i) added.
Cell 7    Optuna tuning for Task B base learners on the augmented feature
          set (with OOF tier features).
Cell 8    Task B stacked ensemble: 6 tuned base + Ridge meta, 5-fold CV.
          CV R^2 = 0.8453 (+/- 0.0140).
Cell 9    Section header.
Cell 10   Final fit on full training data; predict on test; clip rate to
          [4.99, 35.99] and round to 2 decimals; write submission.csv.

---------------------------------------------------------------------
4. Reproducibility notes
---------------------------------------------------------------------

  * random_state=42 is set on every model and every fold split.
  * 5-fold StratifiedKFold for Task A; 5-fold KFold for Task B.
  * Cross-task tier features use cross_val_predict so each row's tier
    probabilities come from a model that did not see that row.
  * Optuna trials are stochastic but converge to the same neighbourhood
    on repeat runs; final stacked CV scores are stable to the third
    decimal place across repeats on the same hardware.
  * Running the notebook from scratch will reproduce
    outputs/submission.csv with identical RiskTier predictions and
    InterestRate predictions within ~0.05 absolute (Optuna noise).

---------------------------------------------------------------------
5. Environment notes
---------------------------------------------------------------------

  * GPU not required.
  * Peak memory ~4 GB.
  * Tested on macOS 14.x, Python 3.13.5, 12-core Apple Silicon.

---------------------------------------------------------------------
6. Known results
---------------------------------------------------------------------

  * Task A (RiskTier classification): 88.12% accuracy, 5-fold CV.
  * Task B (InterestRate regression): R^2 = 0.8453, 5-fold CV.
  * Combined CV score: 0.8633.

  See report Section 3 for the comparison against the alternative
  170-feature pipeline in code/experiments/our_pipeline/, which scores
  0.8407 combined and is kept as a documented alternative branch.
