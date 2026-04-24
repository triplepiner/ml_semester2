"""
iter13.py — feature-bagged ensemble.

Our base models all score 0.79-0.83 and their OOFs correlate strongly
because they see the same feature matrix. To force genuine diversity we
train 3 new "bagged" models, each seeing a random 70% subset of features
with a different random seed. Different features → different splits →
different errors → bigger ensemble lift.

We add these 3 Task-A models + 3 Task-B models to the ensemble, then
rebuild stage-2 and re-blend.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

import lightgbm as lgb

from utils import (ID_COL, OOF_DIR, OUT_DIR, RATE_MAX, RATE_MIN, SEED,
                   TARGET_A, TARGET_B, clip_rate, get_folds, load_data,
                   print_header, set_seed)
from preprocessing import preprocess
from features import engineer_features
from advanced import ridge_blend_classification, ridge_blend_regression
from iter7 import load_latest_oofs, stage2_lgb
from iter12 import build_enriched_features

N_CLASSES = 5


def fit_bagged_lgb_cls(X, y, Xt, folds, feat_idx, seed, rounds=3500):
    Xs_tr = X.iloc[:, feat_idx]; Xs_te = Xt.iloc[:, feat_idx]
    p = dict(objective="multiclass", num_class=N_CLASSES, metric="multi_logloss",
             learning_rate=0.02, num_leaves=95, min_child_samples=20,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
             lambda_l2=1.0, verbose=-1, seed=seed)
    oof = np.zeros((len(Xs_tr), N_CLASSES))
    tp = np.zeros((len(Xs_te), N_CLASSES))
    for tri, vai in folds:
        m = lgb.train(p, lgb.Dataset(Xs_tr.iloc[tri], y.iloc[tri]), rounds,
                      valid_sets=[lgb.Dataset(Xs_tr.iloc[vai], y.iloc[vai])],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        oof[vai] = m.predict(Xs_tr.iloc[vai], num_iteration=m.best_iteration)
        tp += m.predict(Xs_te, num_iteration=m.best_iteration) / len(folds)
    return oof, tp


def fit_bagged_lgb_reg(X, y, Xt, folds, feat_idx, seed, rounds=3500):
    Xs_tr = X.iloc[:, feat_idx]; Xs_te = Xt.iloc[:, feat_idx]
    p = dict(objective="regression", metric="rmse", learning_rate=0.02,
             num_leaves=95, min_child_samples=20, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
             verbose=-1, seed=seed)
    oof = np.zeros(len(Xs_tr)); tp = np.zeros(len(Xs_te))
    for tri, vai in folds:
        m = lgb.train(p, lgb.Dataset(Xs_tr.iloc[tri], y.iloc[tri]), rounds,
                      valid_sets=[lgb.Dataset(Xs_tr.iloc[vai], y.iloc[vai])],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        oof[vai] = m.predict(Xs_tr.iloc[vai], num_iteration=m.best_iteration)
        tp += m.predict(Xs_te, num_iteration=m.best_iteration) / len(folds)
    return oof, tp


def main():
    set_seed(SEED)
    t0 = time.time()
    print_header("iter13 — feature-bagged ensemble")

    X_train_new, X_test_new, y_tier, y_rate, ids_test = build_enriched_features()
    print(f"  enriched X_train={X_train_new.shape}")
    folds = get_folds(y_tier)

    n_feat = X_train_new.shape[1]
    frac = 0.70
    rng = np.random.default_rng(SEED)

    # Make 3 random feature subsets, 3 seeds
    bags = []
    for i, seed in enumerate([42, 2024, 9999]):
        idx = rng.choice(n_feat, size=int(frac * n_feat), replace=False)
        idx.sort()
        bags.append((i, seed, idx))
    print(f"  bags: {[len(idx) for _, _, idx in bags]} features each")

    oof_A_bags, test_A_bags = {}, {}
    oof_B_bags, test_B_bags = {}, {}

    for i, seed, feat_idx in bags:
        print_header(f"Bag {i} (seed {seed}, {len(feat_idx)} features)")
        t = time.time()
        oof, tp = fit_bagged_lgb_cls(X_train_new, y_tier, X_test_new, folds,
                                     feat_idx, seed, rounds=3500)
        acc = accuracy_score(y_tier, oof.argmax(1))
        oof_A_bags[f"bag{i}"] = oof; test_A_bags[f"bag{i}"] = tp
        np.save(OOF_DIR / f"v13_bag{i}_A_oof.npy", oof)
        np.save(OOF_DIR / f"v13_bag{i}_A_test.npy", tp)
        print(f"  A/bag{i} acc={acc:.4f}  ({time.time()-t:.0f}s)")

        t = time.time()
        oof, tp = fit_bagged_lgb_reg(X_train_new, y_rate, X_test_new, folds,
                                     feat_idx, seed, rounds=3500)
        r2 = r2_score(y_rate, oof)
        oof_B_bags[f"bag{i}"] = oof; test_B_bags[f"bag{i}"] = tp
        np.save(OOF_DIR / f"v13_bag{i}_B_oof.npy", oof)
        np.save(OOF_DIR / f"v13_bag{i}_B_test.npy", tp)
        print(f"  B/bag{i} R²={r2:.4f}  ({time.time()-t:.0f}s)")

    # ---- Assemble OOFs ----
    print_header("Assemble latest OOFs + bagged models")
    oof_A, test_A, oof_B, test_B = load_latest_oofs(OOF_DIR, y_tier, y_rate)
    if (OOF_DIR / "v9_cat_A_oof.npy").exists():
        oof_A["cat"] = np.load(OOF_DIR / "v9_cat_A_oof.npy")
        test_A["cat"] = np.load(OOF_DIR / "v9_cat_A_test.npy")
    if (OOF_DIR / "v9_cat_B_oof.npy").exists():
        oof_B["cat"] = np.load(OOF_DIR / "v9_cat_B_oof.npy")
        test_B["cat"] = np.load(OOF_DIR / "v9_cat_B_test.npy")
    if (OOF_DIR / "v12_floor_B_oof.npy").exists():
        oof_B["floor"] = np.load(OOF_DIR / "v12_floor_B_oof.npy")
        test_B["floor"] = np.load(OOF_DIR / "v12_floor_B_test.npy")
    for k, v in oof_A_bags.items():
        oof_A[k] = v; test_A[k] = test_A_bags[k]
    for k, v in oof_B_bags.items():
        oof_B[k] = v; test_B[k] = test_B_bags[k]

    # ---- Stage-2 ----
    print_header("Stage-2 with bagged OOFs")
    parts_tr = [X_train_new.reset_index(drop=True)]
    parts_te = [X_test_new.reset_index(drop=True)]
    for m, arr in oof_A.items():
        cols = [f"oofA_{m}_p{k}" for k in range(arr.shape[1])]
        parts_tr.append(pd.DataFrame(arr, columns=cols))
        parts_te.append(pd.DataFrame(test_A[m], columns=cols))
    try:
        ord_f = np.load(OOF_DIR / "lgb_ord_A_oof_float.npy")
        ord_t = np.load(OOF_DIR / "lgb_ord_A_test_float.npy")
        parts_tr.append(pd.DataFrame({"oofA_ord_float": ord_f}))
        parts_te.append(pd.DataFrame({"oofA_ord_float": ord_t}))
    except FileNotFoundError:
        pass
    for m, arr in oof_B.items():
        parts_tr.append(pd.DataFrame({f"oofB_{m}": arr}))
        parts_te.append(pd.DataFrame({f"oofB_{m}": test_B[m]}))
    X_aug_tr = pd.concat(parts_tr, axis=1)
    X_aug_te = pd.concat(parts_te, axis=1)
    print(f"  X_aug_tr={X_aug_tr.shape}")

    t = time.time()
    s2_A_oof, s2_A_test = stage2_lgb(X_aug_tr, y_tier, X_aug_te, folds, "A")
    print(f"  stage2_A acc={accuracy_score(y_tier, s2_A_oof.argmax(1)):.4f}  ({time.time()-t:.0f}s)")
    t = time.time()
    s2_B_oof, s2_B_test = stage2_lgb(X_aug_tr, y_rate, X_aug_te, folds, "B")
    print(f"  stage2_B  R²={r2_score(y_rate, s2_B_oof):.4f}  ({time.time()-t:.0f}s)")

    oof_A["stack2"] = s2_A_oof; test_A["stack2"] = s2_A_test
    oof_B["stack2"] = s2_B_oof; test_B["stack2"] = s2_B_test

    # ---- Ensemble ----
    print_header("Ensemble")
    from stack import optimise_blend_classification, optimise_blend_regression

    w_A = optimise_blend_classification(oof_A, y_tier.to_numpy())
    fA_conv_oof = sum(w_A[n] * oof_A[n] for n in w_A)
    fA_conv_test = sum(w_A[n] * test_A[n] for n in w_A)
    acc_conv = accuracy_score(y_tier, fA_conv_oof.argmax(1))
    _, rA_oof, rA_test = ridge_blend_classification(oof_A, y_tier.to_numpy(), test_A, alpha=1.0)
    acc_ridge = accuracy_score(y_tier, rA_oof.argmax(1))
    print(f"  A convex acc={acc_conv:.4f}   ridge acc={acc_ridge:.4f}")
    print(f"  A weights: { {k: round(float(v),3) for k,v in w_A.items()} }")
    if acc_ridge > acc_conv:
        final_A_oof, final_A_test = rA_oof, rA_test; acc_final, A_method = acc_ridge, "ridge"
    else:
        final_A_oof, final_A_test = fA_conv_oof, fA_conv_test; acc_final, A_method = acc_conv, "convex"

    w_B = optimise_blend_regression(oof_B, y_rate.to_numpy())
    fB_conv_oof = sum(w_B[n] * oof_B[n] for n in w_B)
    fB_conv_test = sum(w_B[n] * test_B[n] for n in w_B)
    r2_conv = r2_score(y_rate, fB_conv_oof)
    _, _, rB_oof, rB_test = ridge_blend_regression(oof_B, y_rate.to_numpy(), test_B, alpha=1.0)
    r2_ridge = r2_score(y_rate, rB_oof)
    print(f"  B convex R²={r2_conv:.4f}   ridge R²={r2_ridge:.4f}")
    print(f"  B weights: { {k: round(float(v),3) for k,v in w_B.items()} }")
    if r2_ridge > r2_conv:
        final_B_oof, final_B_test = rB_oof, rB_test; r2_final, B_method = r2_ridge, "ridge"
    else:
        final_B_oof, final_B_test = fB_conv_oof, fB_conv_test; r2_final, B_method = r2_conv, "convex"

    combined = 0.5 * acc_final + 0.5 * r2_final
    print_header("iter13 FINAL OOF")
    print(f"  Accuracy       : {acc_final:.4f}  ({A_method})")
    print(f"  R²             : {r2_final:.4f}  ({B_method})")
    print(f"  Combined score : {combined:.4f}")
    print(f"  vs iter8 0.8407 → delta = {combined - 0.8407:+.4f}")

    with open(OOF_DIR / "iter13_weights.json", "w") as f:
        json.dump({"A_method": A_method, "B_method": B_method,
                   "A_convex": {k: float(v) for k, v in w_A.items()},
                   "B_convex": {k: float(v) for k, v in w_B.items()},
                   "acc": acc_final, "r2": r2_final, "combined": combined},
                  f, indent=2)

    # Determine best-so-far
    best = 0.8407
    for j in ["iter9_weights.json", "iter10_weights.json", "iter11_weights.json",
              "iter12_weights.json"]:
        p = OOF_DIR / j
        if p.exists():
            try:
                best = max(best, json.load(open(p))["combined"])
            except Exception:
                pass
    if combined > best:
        tier_pred = final_A_test.argmax(1).astype(int)
        rate_pred = clip_rate(final_B_test)
        sub = pd.DataFrame({ID_COL: ids_test.astype(int).to_numpy(),
                            TARGET_A: tier_pred, TARGET_B: rate_pred})
        sub.to_csv(OUT_DIR / "submission.csv", index=False)
        print(f"  *** iter13 improves over {best:.4f} — submission updated to {combined:.4f} ***")
    else:
        print(f"  iter13 ({combined:.4f}) did not beat {best:.4f}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
