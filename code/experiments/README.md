# experiments/ — additional pipelines and iteration history

These scripts are **not** required to reproduce the submitted CSV. They
document earlier work and parallel attempts, kept for traceability and
context.

## `our_pipeline/` — the heavy-engineering branch

A different approach written by another team member that reached **0.8407**
combined OOF: 170 engineered features, multi-target encoding, K-NN target
features, multi-seed stage-2 LightGBM stacker, pseudo-labeling, adversarial
validation, KNN features, monotone constraints, and 14 documented
iterations. Run with:

```
cd code/experiments/our_pipeline/
python reproduce_final.py
```

Final OOF: 0.8409 (acc 0.8402, R² 0.8416). This is **lower** than the
canonical pipeline because the heavy feature engineering ended up
adding noise — see report Section 2.

Contents:
- `utils.py`, `preprocessing.py`, `features.py` — core modules (~50 features)
- `reproduce_final.py` — single-entry pipeline reproducing the 0.84 OOF
- `01_eda.ipynb` — earlier EDA notebook

## Loose iteration scripts (`iter*.py`, `run_all.py`, etc.)

| Script | Purpose | OOF combined |
|---|---|---|
| `run_all.py`    | Original end-to-end baseline. | 0.8323 |
| `train_base.py` | Initial LGB/XGB/CAT base training helpers. | — |
| `stack.py`      | First stacking + ensemble-weight optimiser. | — |
| `predict.py`    | Initial submission writer. | — |
| `iter3.py` / `iter3b.py` / `iter3c.py` | First stacked attempt with tier-4 mixture-of-experts, monotone constraints, two-stage Task A, multi-target encoding. | 0.8389 |
| `iter4.py`      | Retrain at LR 0.02 with multi-seed stage-2. | 0.8384 |
| `iter5.py`      | CatBoost with native categorical features. | 0.8388 |
| `iter6.py`      | First-round pseudo-labeling on 5 000 high-confidence test rows. | 0.8404 |
| `iter7.py`      | Adversarial validation + feature pruning. | 0.8405 |
| `iter8.py`      | K-nearest-neighbour target features in stage-2. | **0.8407** |
| `iter9.py`      | Noise-matched multi-target encoding + log-rate model + group aggregations. | 0.8389 |
| `iter10.py`     | Second-round pseudo-labeling on iter9 predictions. | 0.8407 |
| `iter11.py`     | Tabular MLP as an additional base learner. | 0.8405 |
| `iter12.py`     | Rate-floor two-stage model for Task B. | 0.8386 |
| `iter13.py`     | Feature-bagged LGB ensemble (3 random subsets). | 0.8397 |
| `iter14.py`     | Multi-seed CatBoost averaging (aborted). | — |
| `advanced.py`   | Shared helpers used by iter scripts. | — |

The canonical pipeline that produces our submitted `submission.csv` lives
one directory up at `code/boosting.ipynb`. Aggressive feature selection
(18 features) plus stacked ensembling on six Optuna-tuned base learners
beat the heavy-engineering branch by +0.022 combined.
