# Plan: Iterative parallel-agent feature-engineering search to ≥0.86 combined CV

## Context

The CreditSense Kaggle project (predict `RiskTier` + `InterestRate`) currently scores **combined = 0.8284** in [boosting/boosting.ipynb](boosting/boosting.ipynb) with a stacked ensemble (5 base learners + meta) on 21 selected features. Goal: **≥ 0.86** (matches the PDF "tuned ensemble" benchmark).

The gap is **+0.032**. Hyperparameter tuning is off-limits per prior user direction; the lever is **the training dataset itself** — feature engineering, feature selection, and combinations of those.

The user wants:
1. **Multiple parallel agents**, each trying a *different* FE strategy.
2. **Iteration with learning** — each wave reads previous wave's results so agents don't repeat mistakes.
3. **Loop until ≥0.86** is achieved.
4. **Reproducible best result** saved on disk.
5. **Use** specifically: Chi-squared test, ANOVA F-test, mutual information, **AIC/BIC stepwise**, RFE, L1/Lasso, plus permutation importance and other techniques. Per the survey of existing notebooks ([advanced.ipynb](advanced.ipynb), [brute_force.ipynb](brute_force.ipynb), [mega_brute_force.ipynb](mega_brute_force.ipynb), [slavs_inspired.ipynb](slavs_inspired.ipynb), [ensemble.ipynb](ensemble.ipynb), [xgboost.ipynb](xgboost.ipynb), [tests/individual_features.ipynb](tests/individual_features.ipynb)), **none** of the listed statistical tests have been tried — they are real gaps.

---

## High-level approach: wave-based parallel search with learning

```
Wave 1: 4 agents in parallel (different strategies)  →  reports + scores
        ↓ orchestrator reads all reports
Wave 2: 3 agents (combining winners, fixing failures) →  reports + scores
        ↓
Wave 3: 2 agents (targeted at weakest task)           →  reports + scores
        ↓
Wave 4 (if needed): 2 agents allowed to *also* tune base-model hyperparams
        ↓
Save best result to boosting/experiments/best/
```

**Stop the loop the moment any wave produces a run with `combined ≥ 0.86`.**

---

## Standardized agent contract

Every experimental agent works in its own subdirectory under `boosting/experiments/wave<N>/<strategy_name>/` and **must produce these three files**:

1. **`pipeline.py`** — module exposing `prepare(train_df, test_df) -> (X_train, y_cls, y_reg, X_test, feature_names)` and `MODEL_NAMES = ['xgb1','xgb2','rf','et','hgb']`.
2. **`result.json`** — schema below; written *after* CV completes.
3. **`report.md`** — 200–500 words: techniques tried, what helped, what hurt, surprises, what to try next. The orchestrator feeds this verbatim to later waves.

`result.json` schema:
```json
{
  "strategy": "<name>",
  "wave": 1,
  "cv_acc": 0.8230,
  "cv_r2": 0.8338,
  "combined": 0.8284,
  "n_features": 21,
  "feature_list": ["..."],
  "techniques_used": ["chi2", "mutual_info", "anova_f", "aic_stepwise"],
  "techniques_dropped": ["polynomial_deg2 — overfit"],
  "runtime_min": 12.4
}
```

### Standard eval harness (every agent uses verbatim)

To keep results comparable, every agent evaluates with the **identical** stacked ensemble from [boosting/boosting.ipynb](boosting/boosting.ipynb) (cells `119850ea` + `7d8da0c2`), 5-fold CV. The orchestrator will write this once into `boosting/experiments/eval_harness.py` and each agent imports it:

```python
from eval_harness import score_dataset
acc, r2, combined = score_dataset(X_train, y_cls, y_reg)
```

This eliminates "they used a different model" as a confounder — only the **dataset** changes between agents.

---

## Wave 1 — four strategies in parallel (4 general-purpose agents)

Each agent gets full access to `credit_train.csv` / `credit_test.csv` and must build its pipeline from scratch (not import `boosting.ipynb`). Budget per agent: 30 minutes.

### Strategy A — `statistical_filter`
- **Chi-squared test** for each label-encoded categorical vs `RiskTier`.
- **ANOVA F-test** (`f_classif` / `f_regression`) for every continuous feature vs each target.
- **Mutual information** (already proven useful — keep as baseline).
- Combine the three rankings via Borda count; `SelectKBest` style with `K ∈ {15, 25, 35, 50}`. Pick the K with the best 5-fold CV.
- Build basic engineered features first (the four already in `boosting.ipynb` plus `BadAccountRatio`, `RecentInquiryShare`, `TotalDebt`, `NetWorthProxy`).

### Strategy B — `info_criteria_stepwise`
- **Forward stepwise** feature selection optimizing **AIC** on a multinomial logistic regression for Task A.
- **Forward stepwise** optimizing **BIC** on linear regression for Task B (BIC penalizes complexity more — should give a leaner, more transferable set).
- Take the union of the two selected sets.
- Rich starting pool: original 55 + ~20 engineered features (interactions on top-5, log1p on skewed >2.0, ratios from `xgboost.ipynb` notebook the survey identified).
- Use `statsmodels` for proper AIC/BIC; or hand-roll if statsmodels too slow.

### Strategy C — `embedded_l1_rfe`
- **L1-regularized** models with CV: `LassoCV` for regression, `LogisticRegressionCV(penalty='l1', solver='saga')` for classification — gives sparse weights → drop zero-weight features.
- **RFE** with XGBoost as estimator (`step=2`, `n_features_to_select` swept).
- Take the **intersection** (most conservative) and **union** (most permissive) and CV both.
- Plus permutation importance as a tie-breaker.

### Strategy D — `aggressive_engineering`
- Generate a **large candidate pool** (~80–120 features): all pairwise products of top-10 by MI; ratios `a/(b+1)` for top-10; `log1p` for skewness >2; `sqrt`, `square` of top-5; `KBinsDiscretizer` for top-3 continuous; **target encoding** (CV-based via `sklearn.preprocessing.TargetEncoder`) for the surviving 4 categoricals.
- Then prune aggressively via **permutation importance** on the stacked ensemble down to top-30.
- This is the "throw a lot at the wall" strategy — most likely to overfit, but if any genuine new signal exists, it'll surface here.

---

## Iteration protocol — how the orchestrator learns

After Wave 1 completes (all 4 `result.json` written), the orchestrator (Claude Code) does:

1. **Read all 4 `result.json` + `report.md`**.
2. **Pick the winner** — best `combined`. If ≥0.86, stop.
3. **Identify what helped**: features appearing in 2+ winning sets are "robust"; techniques flagged "helped" in reports are repeatable.
4. **Identify what failed**: features universally rejected; techniques flagged "overfit" or "no signal".
5. **Compose Wave 2 agent prompts** with explicit "Lessons from Wave 1" section embedded in each prompt. Wave 2 agents are explicitly forbidden from re-trying failed approaches.

### Wave 2 — three agents (combine + extend)

- **`E — winners_union`**: union of top-2 wave-1 feature sets; resolve duplicates by signal score.
- **`F — target_encoding_focus`**: take the wave-1 winner's set and *only* upgrade categorical encoding (CV target encoding with smoothing, leave-one-out, M-estimate) — measure the lift from encoding alone.
- **`G — interaction_search`**: take the wave-1 winner's set and add **only** interactions found via decision-tree path mining (fit a small tree, extract pairs that frequently co-split).

### Wave 3 (only if `combined < 0.86`)

Look at the score breakdown. If accuracy < 0.85, the bottleneck is Task A — agent **`H — class_balanced_resampling`** trains the classifier with `sample_weight` derived from per-class error analysis on OOF predictions. If R² < 0.86, the bottleneck is Task B — agent **`I — floor_aware_regressor`** trains a 2-stage model (binary "is rate at 4.99 floor?" + regressor on non-floor rows), since the EDA showed ~25% mass piled at the floor.

### Wave 4 — last resort, allow tuning

If Wave 3 still doesn't crack 0.86, two agents are permitted to do **light** hyperparameter search (Optuna ≤30 trials, single-task) on top of the best feature set found so far. This violates the "no tuning" preference but is the documented escape hatch. The orchestrator must ask the user to authorize Wave 4 before launching.

---

## Saving the best result — reproducibility

Best run lands in `boosting/experiments/best/`:

```
best/
├── pipeline.py         # the winning prepare() function
├── eval_harness.py     # copy of the standard harness
├── train_and_save.py   # script: load data → prepare → fit stacked → save model
├── model.pkl           # joblib-pickled fitted StackingClassifier + StackingRegressor
├── submission.csv      # predictions on credit_test.csv
├── result.json         # final scores + provenance (which wave/strategy)
├── feature_list.txt    # one feature per line
├── requirements.txt    # exact package versions used
└── README.md           # 1-page reproduction recipe:
                        #   1. pip install -r requirements.txt
                        #   2. python train_and_save.py
                        #   3. inspect submission.csv
                        # + reports the achieved CV score
```

Random seeds fixed (`random_state=42`) everywhere. The `train_and_save.py` script must produce **bit-identical** `submission.csv` on re-run.

---

## Critical files — to be created

- `boosting/experiments/eval_harness.py` *(new — orchestrator writes once before Wave 1)*
- `boosting/experiments/wave1/{statistical_filter,info_criteria_stepwise,embedded_l1_rfe,aggressive_engineering}/` *(4 dirs, agent-written)*
- `boosting/experiments/wave2/...` *(if needed)*
- `boosting/experiments/wave3/...` *(if needed)*
- `boosting/experiments/best/` *(final, after winner found)*

**Untouched**: [boosting/boosting.ipynb](boosting/boosting.ipynb) stays as-is — it's the baseline reference. Existing notebooks ([eda.ipynb](eda.ipynb), [advanced.ipynb](advanced.ipynb), etc.) are read-only references for the agents.

---

## Verification — how to confirm success end-to-end

1. After Wave-N completes, read every `result.json` and confirm at least one has `combined ≥ 0.86`.
2. Run `python boosting/experiments/best/train_and_save.py` from a clean shell — verify it completes without error and produces `submission.csv` matching the saved one (md5 check).
3. Re-load `model.pkl` and re-score on a held-out 20% slice — verify the score is consistent with reported CV (within ±0.005).
4. Inspect `submission.csv` schema: 15,000 rows, columns `Id,RiskTier,InterestRate`, RiskTier ∈ {0..4}, InterestRate ∈ [4.99, 35.99].

---

## Stop conditions

- ✅ Any agent reports `combined ≥ 0.86` → save best, exit.
- ⚠️ End of Wave 3 with no agent ≥ 0.86 → ask user to authorize Wave 4 (light hyperparameter tuning).
- ❌ End of Wave 4 still below 0.86 → save best-so-far, write a `BLOCKED.md` documenting the gap and what the bottleneck appears to be (most likely the model, not the data).

---

## Risks & honest caveats

- **The 0.86 ceiling may not be reachable from FE alone.** The PDF's "tuned ensemble" benchmark of 0.83–0.85 was *with* hyperparameter tuning. The survey shows existing tuned notebooks max at ~0.833. Beating 0.86 may require Wave 4 (tuning).
- **Compute budget**: 4 parallel agents × 5-fold CV with 5-base-learner stacking ≈ 20–40 min/wave. Total: 1–3 hours wall-clock if all 4 waves run.
- **Overfitting CV**: with 4 strategies × multiple K choices each, we're search-fitting against the same 5-fold split. The winning strategy's CV score may be optimistic by ~0.005. The Kaggle public leaderboard is the only honest tiebreaker.
