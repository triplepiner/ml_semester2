# experiments/ — iteration history

These 15 scripts are **not** needed to reproduce the submitted CSV. They are
kept verbatim to document the iteration ladder summarised in
`report/CreditSense_Report.pdf` §3.

| Script | Purpose | OOF combined |
|---|---|---|
| `run_all.py`    | Original end-to-end baseline pipeline. | 0.8323 |
| `train_base.py` | Initial LGB/XGB/CAT base training helpers used by `run_all.py`. | — |
| `stack.py`      | First stacking + ensemble-weight optimiser (convex only). | — |
| `predict.py`    | Initial submission writer. | — |
| `iter3.py` / `iter3b.py` / `iter3c.py` | First stacked attempt with tier-4 mixture-of-experts, monotone constraints, two-stage Task A, multi-target encoding. | 0.8389 |
| `iter4.py`      | Retrain at LR 0.02 with multi-seed stage-2. | 0.8384 |
| `iter5.py`      | CatBoost with native categorical features. | 0.8388 |
| `iter6.py`      | First-round pseudo-labeling on 5 000 high-confidence test rows. | 0.8404 |
| `iter7.py`      | Adversarial validation + feature pruning. | 0.8405 |
| `iter8.py`      | **Champion.** K-nearest-neighbour target features in stage-2. | **0.8407** |
| `iter9.py`      | Noise-matched multi-target encoding + log-rate model + group aggregations. | 0.8389 |
| `iter10.py`     | Second-round pseudo-labeling on iter9 predictions. | 0.8407 |
| `iter11.py`     | Tabular MLP as an additional base learner. | 0.8405 |
| `iter12.py`     | Rate-floor two-stage model for Task B. | 0.8386 |
| `iter13.py`     | Feature-bagged LGB ensemble (3 random subsets). | 0.8397 |
| `iter14.py`     | Multi-seed CatBoost averaging (did not finish — aborted). | — |
| `advanced.py`   | Shared helpers for the later iterations: monotone maps, multi-TE, mixture-of-experts, two-stage, DART, Ridge/Lasso meta-blenders. | — |

The canonical pipeline that reproduces the submitted `submission.csv` lives
one directory up at `code/reproduce_final.py` and is entirely self-contained
(it does not import from `experiments/`).
