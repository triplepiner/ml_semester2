CreditSense — Loan Risk Assessment Challenge (AI1215, Spring 2026)
====================================================================

Team name     : Slavs
Team members  : Makar Ulesov, Ivan Kanev, Delyan Hristov
Kaggle user(s): <KAGGLE_USERNAME>      # fill in before submission

--------------------------------------------------------------------
1. Reproducing the submitted Kaggle CSV
--------------------------------------------------------------------

Prerequisites
  * Python 3.10, 3.11, 3.12 or 3.13 (tested on 3.13.5)
  * ~8 GB free RAM, ~500 MB disk for outputs

Step 1 - download data
  Join the competition and download credit_train.csv + credit_test.csv
      https://www.kaggle.com/t/3e62a127eb85418aa851a5ee258e7c04
  Place both CSVs inside  ml_final/data/  (create the folder if missing).

Step 2 - install dependencies
      python -m venv .venv
      source .venv/bin/activate            # Windows: .venv\Scripts\activate
      pip install -r requirements.txt

Step 3 - run the pipeline
      cd code/
      python reproduce_final.py

  This single script is the canonical reproduction entry point. It runs
  the full pipeline end-to-end from the raw CSVs, prints per-phase
  diagnostics, and writes outputs/submission.csv.

  Expected runtime: 60-80 minutes on an 8-core CPU.
  Expected combined OOF score: ~0.84 (0.5*Accuracy + 0.5*R^2).
  All randomness is seeded via utils.set_seed (SEED=42).

Step 4 - submit to Kaggle
  Upload  outputs/submission.csv  to the competition page.

--------------------------------------------------------------------
2. File map
--------------------------------------------------------------------

  data/
    credit_train.csv         - downloaded from Kaggle
    credit_test.csv          - downloaded from Kaggle
  code/
    utils.py                 - seeds, CV splits, scoring, target encoder
    preprocessing.py         - missing indicators, imputation, encoding
    features.py              - ~50 engineered features with financial intuition
    reproduce_final.py       - *** canonical entry point ***
    01_eda.ipynb             - exploratory data analysis (feeds report Sec.1)
    experiments/             - historical iteration scripts (iter3..iter14,
                               run_all, train_base, stack, predict, advanced).
                               Kept for traceability — not required for the
                               official reproduction. See report Section 3
                               for the iteration ladder and ablation table.
  outputs/
    oof/                     - per-model out-of-fold + test prediction arrays
    submission.csv           - Kaggle upload produced by reproduce_final.py
  report/
    CreditSense_Report.pdf   - 5-page written report
  final_notebook.ipynb       - Colab-runnable consolidated notebook
  README.txt
  requirements.txt

--------------------------------------------------------------------
3. Pipeline overview (reproduce_final.py)
--------------------------------------------------------------------

Phase 1 - Load raw data, engineer ~50 features (balance-sheet ratios,
          delinquency severity, credit-history maturity, interactions,
          log transforms), preprocess (missing indicators, imputation,
          winsorisation, categorical encoding, multi-target encoding).

Phase 2 - Train seven base learners at learning_rate=0.02 with early
          stopping on 5-fold StratifiedKFold:
              LGB / XGB / CatBoost classifiers  (Task A: RiskTier)
              LGB / XGB / CatBoost regressors   (Task B: InterestRate)
              LGB regression-on-tier            (ordinal trick)
              Binary-then-4-class cascade       (two-stage Task A)

Phase 3 - Generate pseudo-labels for the 5000 highest-confidence test
          rows (Task A max probability >= 0.90).

Phase 4 - Retrain the six LGB/XGB/CAT base learners on the enlarged
          40 000-row training set (35 000 real + 5 000 pseudo).

Phase 5 - Compute K=10 nearest-neighbour target features per row
          (mean rate, std rate, tier distribution).

Phase 6 - Stage-2 meta-learner: multi-seed LightGBM (seeds 42, 1337,
          2024) trained on all base-model OOF predictions + KNN features
          + original features. Averaged across seeds.

Phase 7 - Ensemble blending: both convex (Nelder-Mead on log-loss /
          negative R^2) and Ridge meta-blender. Higher OOF winner
          selected per task.

Phase 8 - Validate + write submission.csv (Id, RiskTier, InterestRate).

--------------------------------------------------------------------
4. Reproducibility guarantees
--------------------------------------------------------------------

  * Global seed SEED = 42 set in code/utils.py (Python random, NumPy,
    PYTHONHASHSEED). Every model receives this seed explicitly.
  * 5-fold StratifiedKFold on RiskTier; the same fold indices are
    reused for Task B so out-of-fold predictions align across tasks.
  * K-fold target encoding (fold i encoded using folds != i) prevents
    leakage from InterestRate into training features.
  * OOF evaluation for pseudo-labeled rows is suppressed (metrics
    reported on the real 35 000 rows only) to avoid over-estimating
    performance.
  * Running python code/reproduce_final.py on the same machine with
    identical library versions reproduces the submission.csv we
    uploaded to Kaggle. Trivial floating-point drift across operating
    systems may cause differences in the 3rd-4th decimal place but
    does not change RiskTier predictions.

--------------------------------------------------------------------
5. Environment notes
--------------------------------------------------------------------

  * GPU not required. LightGBM / XGBoost tree_method='hist' and
    CatBoost default CPU mode all fit 50 000 * ~180 features in ~10-20
    seconds per fold each.
  * Peak memory ~2 GB. Runs comfortably on a consumer laptop.
  * Tested on: macOS 14.x, Python 3.13.5, 12-core Apple Silicon.

--------------------------------------------------------------------
6. Known results on OOF
--------------------------------------------------------------------

  * Phase 2 base models cluster: acc 0.79-0.81 (Task A), R^2 0.83-0.84.
  * Phase 6 stage-2 meta-learner: acc ~0.84, R^2 ~0.84.
  * Phase 7 final blend: combined score ~0.84.

  The iteration ladder in experiments/ (iter3 through iter14) documents
  techniques attempted on top of this baseline: tier-4 mixture-of-experts,
  monotone constraints, DART boosting, adversarial-validation feature
  pruning, second-round pseudo-labeling, feature bagging, multi-seed
  CatBoost averaging, and others. None moved the combined score above
  0.841 - our investigation of this ceiling is reported in the written
  report Section 3.
