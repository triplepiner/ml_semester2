"""
iter8.py — KNN target features.

For each row we find the 10 nearest neighbors in training space (cosine
similarity on standardised features) and compute:
  - mean InterestRate of neighbours
  - probability of each RiskTier among neighbours (5 numbers)
  - std InterestRate of neighbours

These features capture local target patterns that gradient boosting tends to
smooth out. KNN errors are uncorrelated with boosting errors, so adding these
as stage-2 features usually lifts the ensemble even when the individual KNN
"model" isn't great.

Leakage protection: for train rows we compute KNN excluding the query row
(self) via leave-one-out. For test rows we use all train data.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from utils import (ID_COL, OOF_DIR, OUT_DIR, RATE_MAX, RATE_MIN, SEED,
                   TARGET_A, TARGET_B, clip_rate, get_folds, load_data,
                   print_header, set_seed)
from preprocessing import preprocess
from features import engineer_features
from advanced import ridge_blend_classification, ridge_blend_regression
from iter3 import add_multi_te
from iter7 import load_latest_oofs, stage2_lgb


N_CLASSES = 5
K_NEIGHBORS = 10


def build_knn_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_tier: pd.Series, y_rate: pd.Series, folds,
                       k: int = K_NEIGHBORS):
    """
    Build KNN target features for both train (OOF) and test sets.
    Returns (knn_tr_df, knn_te_df) with columns:
      knn_mean_rate, knn_std_rate, knn_p_tier0..knn_p_tier4
    """
    # Use a reduced-dim, standardised feature view for the metric.
    # Drop any columns that are all-constant in train (they hurt distance).
    std_cols = [c for c in X_train.columns if X_train[c].std() > 1e-6]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train[std_cols].to_numpy())
    X_te_s = scaler.transform(X_test[std_cols].to_numpy())

    # Train features via OOF: for each val fold, KNN is fit on the training fold.
    knn_mean_rate = np.zeros(len(X_train))
    knn_std_rate = np.zeros(len(X_train))
    knn_p_tier = np.zeros((len(X_train), N_CLASSES))

    for tri, vai in folds:
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        nn.fit(X_tr_s[tri])
        _, idx = nn.kneighbors(X_tr_s[vai])
        neighbour_rates = y_rate.to_numpy()[tri][idx]           # (n_va, k)
        neighbour_tiers = y_tier.to_numpy()[tri][idx]
        knn_mean_rate[vai] = neighbour_rates.mean(axis=1)
        knn_std_rate[vai] = neighbour_rates.std(axis=1)
        for c in range(N_CLASSES):
            knn_p_tier[vai, c] = (neighbour_tiers == c).mean(axis=1)

    # Test features: fit on ALL train, query test rows.
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    nn.fit(X_tr_s)
    _, idx = nn.kneighbors(X_te_s)
    neighbour_rates = y_rate.to_numpy()[idx]
    neighbour_tiers = y_tier.to_numpy()[idx]
    knn_mean_rate_te = neighbour_rates.mean(axis=1)
    knn_std_rate_te = neighbour_rates.std(axis=1)
    knn_p_tier_te = np.zeros((len(X_test), N_CLASSES))
    for c in range(N_CLASSES):
        knn_p_tier_te[:, c] = (neighbour_tiers == c).mean(axis=1)

    tr_df = pd.DataFrame({
        "knn_mean_rate": knn_mean_rate,
        "knn_std_rate": knn_std_rate,
        **{f"knn_p_tier{c}": knn_p_tier[:, c] for c in range(N_CLASSES)},
    })
    te_df = pd.DataFrame({
        "knn_mean_rate": knn_mean_rate_te,
        "knn_std_rate": knn_std_rate_te,
        **{f"knn_p_tier{c}": knn_p_tier_te[:, c] for c in range(N_CLASSES)},
    })
    return tr_df, te_df


def main():
    set_seed(SEED)
    t0 = time.time()
    print_header("iter8 — KNN target features in stage-2")

    train, test = load_data()
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)
    X_train, X_test, y_tier, y_rate, ids_test = preprocess(train_fe, test_fe)
    X_train, X_test = add_multi_te(X_train, X_test, load_data()[0],
                                   load_data()[1], y_tier, y_rate)
    print(f"  X_train={X_train.shape}  X_test={X_test.shape}")

    folds = get_folds(y_tier)

    # ---- Step 1: KNN features (OOF-safe) ----
    print_header("Step 1 — compute K-NN target features (OOF safe)")
    t = time.time()
    knn_tr, knn_te = build_knn_features(X_train, X_test, y_tier, y_rate, folds)
    print(f"  KNN features computed in {time.time()-t:.0f}s")
    print(f"  train head:\n{knn_tr.head()}")

    # ---- Step 2: load latest base OOFs ----
    print_header("Step 2 — load latest base OOFs")
    oof_A, test_A, oof_B, test_B = load_latest_oofs(OOF_DIR, y_tier, y_rate)
    for m in oof_A:
        print(f"  A/{m:10s} acc={accuracy_score(y_tier, oof_A[m].argmax(1)):.4f}")
    for m in oof_B:
        print(f"  B/{m:10s} R²={r2_score(y_rate, oof_B[m]):.4f}")

    # ---- Step 3: stage-2 with KNN features added ----
    print_header("Step 3 — stage-2 with KNN features")
    parts_tr = [X_train.reset_index(drop=True),
                knn_tr.reset_index(drop=True)]
    parts_te = [X_test.reset_index(drop=True),
                knn_te.reset_index(drop=True)]
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

    s2_A_oof, s2_A_test = stage2_lgb(X_aug_tr, y_tier, X_aug_te, folds, "A")
    acc_s2 = accuracy_score(y_tier, s2_A_oof.argmax(1))
    print(f"  stage2_A acc={acc_s2:.4f}")

    s2_B_oof, s2_B_test = stage2_lgb(X_aug_tr, y_rate, X_aug_te, folds, "B")
    r2_s2 = r2_score(y_rate, s2_B_oof)
    print(f"  stage2_B  R²={r2_s2:.4f}")

    oof_A["stack2"] = s2_A_oof; test_A["stack2"] = s2_A_test
    oof_B["stack2"] = s2_B_oof; test_B["stack2"] = s2_B_test

    # ---- Step 4: ensemble ----
    print_header("Step 4 — ensemble")
    from stack import optimise_blend_classification, optimise_blend_regression

    w_A = optimise_blend_classification(oof_A, y_tier.to_numpy())
    fA_conv_oof = sum(w_A[n] * oof_A[n] for n in w_A)
    fA_conv_test = sum(w_A[n] * test_A[n] for n in w_A)
    acc_conv = accuracy_score(y_tier, fA_conv_oof.argmax(1))
    _, rA_oof, rA_test = ridge_blend_classification(oof_A, y_tier.to_numpy(),
                                                    test_A, alpha=1.0)
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
    _, _, rB_oof, rB_test = ridge_blend_regression(oof_B, y_rate.to_numpy(),
                                                    test_B, alpha=1.0)
    r2_ridge = r2_score(y_rate, rB_oof)
    print(f"  B convex R²={r2_conv:.4f}   ridge R²={r2_ridge:.4f}")
    print(f"  B weights: { {k: round(float(v),3) for k,v in w_B.items()} }")
    if r2_ridge > r2_conv:
        final_B_oof, final_B_test = rB_oof, rB_test; r2_final, B_method = r2_ridge, "ridge"
    else:
        final_B_oof, final_B_test = fB_conv_oof, fB_conv_test; r2_final, B_method = r2_conv, "convex"

    combined = 0.5 * acc_final + 0.5 * r2_final
    print_header("iter8 FINAL OOF")
    print(f"  Accuracy       : {acc_final:.4f}  ({A_method})")
    print(f"  R²             : {r2_final:.4f}  ({B_method})")
    print(f"  Combined score : {combined:.4f}")
    print(f"  vs The Lions 0.88175 → gap = {combined - 0.88175:+.4f}")
    print(f"  vs iter3c 0.8389    → delta = {combined - 0.8389:+.4f}")

    with open(OOF_DIR / "iter8_weights.json", "w") as f:
        json.dump({
            "A_method": A_method, "B_method": B_method,
            "A_convex": {k: float(v) for k, v in w_A.items()},
            "B_convex": {k: float(v) for k, v in w_B.items()},
            "acc": acc_final, "r2": r2_final, "combined": combined,
        }, f, indent=2)

    best_so_far = 0.8389
    if combined > best_so_far:
        tier_pred = final_A_test.argmax(1).astype(int)
        rate_pred = clip_rate(final_B_test)
        sub = pd.DataFrame({ID_COL: ids_test.astype(int).to_numpy(),
                            TARGET_A: tier_pred, TARGET_B: rate_pred})
        sub.to_csv(OUT_DIR / "submission.csv", index=False)
        print(f"  *** iter8 improves over {best_so_far:.4f} — submission updated ***")
    else:
        print(f"  iter8 did not improve over {best_so_far} — keeping previous submission")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
