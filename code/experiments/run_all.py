"""
run_all.py
End-to-end pipeline entry point. Runs preprocessing → feature engineering →
base model training → stacking → submission file.

Usage:
  python run_all.py              # full run, ~45-90 min depending on hardware
  python run_all.py --quick      # fast smoke test on 3k-row subsample
  python run_all.py --tune       # enable Optuna tuning for LightGBM models
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from features import engineer_features
from predict import make_submission
from preprocessing import preprocess
from stack import run_stacking
from train_base import run_all_base
from utils import (OOF_DIR, OUT_DIR, SEED, load_data, print_header, set_seed)


def main(quick: bool = False, tune: bool = False, n_trials: int = 40):
    set_seed(SEED)
    t_start = time.time()

    print_header("Step 1 — Load + engineer features")
    train, test = load_data()
    print(f"  train={train.shape}  test={test.shape}")

    if quick:
        # Sub-sample for fast iteration. Keeps class balance via stratified
        # sampling on RiskTier.
        sub = (train.groupby("RiskTier", group_keys=False)
               .apply(lambda g: g.sample(min(len(g), 600), random_state=SEED)))
        train = sub.reset_index(drop=True)
        test = test.iloc[:3000].reset_index(drop=True)
        print(f"  [quick mode] train={train.shape}  test={test.shape}")

    train_fe = engineer_features(train)
    test_fe = engineer_features(test)
    print(f"  after FE: train={train_fe.shape}  test={test_fe.shape}")

    print_header("Step 2 — Preprocess (impute, encode, winsorize)")
    X_train, X_test, y_tier, y_rate, ids_test = preprocess(train_fe, test_fe)
    print(f"  X_train={X_train.shape}  X_test={X_test.shape}")

    print_header("Step 3 — Base models (LGB / XGB / CAT × tasks A,B)")
    base_scores = run_all_base(X_train, X_test, y_tier, y_rate,
                               tune=tune, n_trials=n_trials, quick=quick)
    print(f"  base scores: {base_scores}")

    print_header("Step 4 — Cross-task stacking + ensemble weights")
    result = run_stacking(X_train, X_test, y_tier, y_rate, quick=quick)

    print_header("Step 5 — Write submission.csv")
    make_submission(ids_test, result["final_test_A"], result["final_test_B"])

    print_header("Pipeline complete")
    print(f"  total time : {(time.time()-t_start)/60:.1f} min")
    print(f"  OOF acc    : {result['acc']:.4f}")
    print(f"  OOF R²     : {result['r2']:.4f}")
    print(f"  Combined   : {result['combined']:.4f}  (Kaggle estimate)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test mode")
    parser.add_argument("--tune", action="store_true", help="Optuna tuning")
    parser.add_argument("--n-trials", type=int, default=40)
    args = parser.parse_args()
    main(quick=args.quick, tune=args.tune, n_trials=args.n_trials)
