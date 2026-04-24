"""
iter14.py — multi-seed CatBoost averaging.

CatBoost is our best single model (0.808 Task A, 0.834 Task B). Same model
with different CV seeds usually produces meaningfully different OOF arrays.
Averaging them reduces variance — the classic bagging gain. Expected
+0.002-0.005 combined on top of a single CAT OOF. Adds the averaged
predictions as a new ensemble member.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold

from utils import (ID_COL, OOF_DIR, OUT_DIR, RATE_MAX, RATE_MIN, SEED,
                   TARGET_A, TARGET_B, clip_rate, load_data, print_header,
                   set_seed)
from preprocessing import preprocess
from features import engineer_features
from advanced import ridge_blend_classification, ridge_blend_regression
from iter7 import load_latest_oofs, stage2_lgb
from iter9 import fit_cat_cls, fit_cat_reg
from iter12 import build_enriched_features


N_CLASSES = 5


def main():
    set_seed(SEED)
    t0 = time.time()
    print_header("iter14 — multi-seed CatBoost averaging")

    X_train_new, X_test_new, y_tier, y_rate, ids_test = build_enriched_features()
    print(f"  enriched X_train={X_train_new.shape}")

    # Train CAT with 2 alternative seeds (original seed 42 already on disk as v9)
    seeds = [2025, 9999]
    oof_A_seeds = []
    test_A_seeds = []
    oof_B_seeds = []
    test_B_seeds = []

    # Include the original v9 CAT as seed 42
    if (OOF_DIR / "v9_cat_A_oof.npy").exists():
        oof_A_seeds.append(np.load(OOF_DIR / "v9_cat_A_oof.npy"))
        test_A_seeds.append(np.load(OOF_DIR / "v9_cat_A_test.npy"))
        oof_B_seeds.append(np.load(OOF_DIR / "v9_cat_B_oof.npy"))
        test_B_seeds.append(np.load(OOF_DIR / "v9_cat_B_test.npy"))
        print("  loaded v9 CAT (seed 42)")

    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        folds_s = list(skf.split(np.arange(len(X_train_new)), y_tier))
        print_header(f"CAT seed {seed}")
        t = time.time()
        oof_A, test_A = fit_cat_cls(X_train_new, y_tier, X_test_new, folds_s, iters=3000)
        oof_A_seeds.append(oof_A); test_A_seeds.append(test_A)
        print(f"  A acc={accuracy_score(y_tier, oof_A.argmax(1)):.4f}  ({time.time()-t:.0f}s)")
        np.save(OOF_DIR / f"v14_cat_A_oof_s{seed}.npy", oof_A)
        np.save(OOF_DIR / f"v14_cat_A_test_s{seed}.npy", test_A)

        t = time.time()
        oof_B, test_B = fit_cat_reg(X_train_new, y_rate, X_test_new, folds_s, iters=3000)
        oof_B_seeds.append(oof_B); test_B_seeds.append(test_B)
        print(f"  B R²={r2_score(y_rate, oof_B):.4f}  ({time.time()-t:.0f}s)")
        np.save(OOF_DIR / f"v14_cat_B_oof_s{seed}.npy", oof_B)
        np.save(OOF_DIR / f"v14_cat_B_test_s{seed}.npy", test_B)

    # Average across seeds
    cat_A_avg_oof = np.mean(oof_A_seeds, axis=0)
    cat_A_avg_test = np.mean(test_A_seeds, axis=0)
    cat_B_avg_oof = np.mean(oof_B_seeds, axis=0)
    cat_B_avg_test = np.mean(test_B_seeds, axis=0)
    print_header("Averaged CAT across seeds")
    print(f"  A acc={accuracy_score(y_tier, cat_A_avg_oof.argmax(1)):.4f}")
    print(f"  B R²={r2_score(y_rate, cat_B_avg_oof):.4f}")
    np.save(OOF_DIR / "v14_cat_A_oof_avg.npy", cat_A_avg_oof)
    np.save(OOF_DIR / "v14_cat_A_test_avg.npy", cat_A_avg_test)
    np.save(OOF_DIR / "v14_cat_B_oof_avg.npy", cat_B_avg_oof)
    np.save(OOF_DIR / "v14_cat_B_test_avg.npy", cat_B_avg_test)

    # ---- Assemble OOFs, use avg CAT as main, keep others ----
    oof_A, test_A, oof_B, test_B = load_latest_oofs(OOF_DIR, y_tier, y_rate)
    oof_A["cat_avg"] = cat_A_avg_oof; test_A["cat_avg"] = cat_A_avg_test
    oof_B["cat_avg"] = cat_B_avg_oof; test_B["cat_avg"] = cat_B_avg_test
    if (OOF_DIR / "v12_floor_B_oof.npy").exists():
        oof_B["floor"] = np.load(OOF_DIR / "v12_floor_B_oof.npy")
        test_B["floor"] = np.load(OOF_DIR / "v12_floor_B_test.npy")
    for k in ["bag0", "bag1", "bag2"]:
        if (OOF_DIR / f"v13_{k}_A_oof.npy").exists():
            oof_A[k] = np.load(OOF_DIR / f"v13_{k}_A_oof.npy")
            test_A[k] = np.load(OOF_DIR / f"v13_{k}_A_test.npy")
        if (OOF_DIR / f"v13_{k}_B_oof.npy").exists():
            oof_B[k] = np.load(OOF_DIR / f"v13_{k}_B_oof.npy")
            test_B[k] = np.load(OOF_DIR / f"v13_{k}_B_test.npy")

    # Use default fold split for stage-2 (seed 42)
    from utils import get_folds
    folds = get_folds(y_tier)

    # ---- Stage-2 ----
    print_header("Stage-2 with everything")
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
    print_header("iter14 FINAL OOF")
    print(f"  Accuracy       : {acc_final:.4f}  ({A_method})")
    print(f"  R²             : {r2_final:.4f}  ({B_method})")
    print(f"  Combined score : {combined:.4f}")
    print(f"  vs iter8 0.8407 → delta = {combined - 0.8407:+.4f}")

    with open(OOF_DIR / "iter14_weights.json", "w") as f:
        json.dump({"A_method": A_method, "B_method": B_method,
                   "A_convex": {k: float(v) for k, v in w_A.items()},
                   "B_convex": {k: float(v) for k, v in w_B.items()},
                   "acc": acc_final, "r2": r2_final, "combined": combined},
                  f, indent=2)

    best = 0.8407
    for j in ["iter9_weights.json", "iter10_weights.json", "iter11_weights.json",
              "iter12_weights.json", "iter13_weights.json"]:
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
        print(f"  *** iter14 improves over {best:.4f} — submission updated to {combined:.4f} ***")
    else:
        print(f"  iter14 ({combined:.4f}) did not beat {best:.4f}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
