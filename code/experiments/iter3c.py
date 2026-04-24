"""
iter3c.py — Resume from where iter3b crashed. 4 Task B models on disk
(lgb_l1, lgb_mono, xgb, cat). Only train: tier-4 mixture + DART, then
stage-2 + final ensemble + submission.
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
from advanced import (train_tier4_mixture, train_dart_reg,
                      ridge_blend_classification, ridge_blend_regression)
from iter3 import add_multi_te

N_CLASSES = 5


def main():
    set_seed(SEED)
    t0 = time.time()
    print_header("iter3c — resume from mixture + DART")

    train_fe = engineer_features(load_data()[0])
    test_fe = engineer_features(load_data()[1])
    train_raw, test_raw = load_data()
    X_train, X_test, y_tier, y_rate, ids_test = preprocess(train_fe, test_fe)
    X_train, X_test = add_multi_te(X_train, X_test, train_raw, test_raw,
                                   y_tier, y_rate)
    folds = get_folds(y_tier)

    # --- Load all saved OOFs ---
    oof_A, test_A = {}, {}
    for m in ["lgb", "xgb", "cat", "lgb_ord", "lgb_mono", "two_stage"]:
        oof_A[m] = np.load(OOF_DIR / f"{m}_A_oof.npy")
        test_A[m] = np.load(OOF_DIR / f"{m}_A_test.npy")
        acc = accuracy_score(y_tier, oof_A[m].argmax(1))
        print(f"  A/{m:10s} acc={acc:.4f}")

    oof_B, test_B = {}, {}
    for m in ["lgb_l1", "lgb_mono", "xgb", "cat"]:
        oof_B[m] = np.load(OOF_DIR / f"{m}_B_oof.npy")
        test_B[m] = np.load(OOF_DIR / f"{m}_B_test.npy")
        r2 = r2_score(y_rate, oof_B[m])
        print(f"  B/{m:10s} R²={r2:.4f}")

    # --- Mixture (fixed free_raw_data bug) ---
    print_header("Tier-4 mixture-of-experts")
    t = time.time()
    oof_B["mix"], test_B["mix"] = train_tier4_mixture(
        X_train, y_rate, y_tier, X_test, folds, rounds=2500)
    np.save(OOF_DIR / "mix_B_oof.npy", oof_B["mix"])
    np.save(OOF_DIR / "mix_B_test.npy", test_B["mix"])
    print(f"  mix R²={r2_score(y_rate, oof_B['mix']):.4f}  ({time.time()-t:.0f}s)")

    # --- DART ---
    print_header("DART regressor (diversity)")
    t = time.time()
    oof_B["dart"], test_B["dart"] = train_dart_reg(X_train, y_rate, X_test, folds,
                                                    rounds=2000)
    np.save(OOF_DIR / "dart_B_oof.npy", oof_B["dart"])
    np.save(OOF_DIR / "dart_B_test.npy", test_B["dart"])
    print(f"  dart R²={r2_score(y_rate, oof_B['dart']):.4f}  ({time.time()-t:.0f}s)")

    # --- Stage-2 ---
    print_header("Stage-2 stacking")
    parts_tr = [X_train.reset_index(drop=True)]
    parts_te = [X_test.reset_index(drop=True)]
    for m, arr in oof_A.items():
        cols = [f"oofA_{m}_p{k}" for k in range(arr.shape[1])]
        parts_tr.append(pd.DataFrame(arr, columns=cols))
        parts_te.append(pd.DataFrame(test_A[m], columns=cols))
    ord_f = np.load(OOF_DIR / "lgb_ord_A_oof_float.npy")
    ord_t = np.load(OOF_DIR / "lgb_ord_A_test_float.npy")
    parts_tr.append(pd.DataFrame({"oofA_ord_float": ord_f}))
    parts_te.append(pd.DataFrame({"oofA_ord_float": ord_t}))
    for m, arr in oof_B.items():
        parts_tr.append(pd.DataFrame({f"oofB_{m}": arr}))
        parts_te.append(pd.DataFrame({f"oofB_{m}": test_B[m]}))
    X_aug_tr = pd.concat(parts_tr, axis=1)
    X_aug_te = pd.concat(parts_te, axis=1)
    print(f"  X_aug_tr={X_aug_tr.shape}")

    p_cls = dict(objective="multiclass", num_class=N_CLASSES,
                 metric="multi_logloss", learning_rate=0.03, num_leaves=31,
                 min_child_samples=30, feature_fraction=0.8,
                 bagging_fraction=0.85, bagging_freq=5, lambda_l2=1.0,
                 verbose=-1, seed=SEED)
    p_reg = dict(objective="regression", metric="rmse", learning_rate=0.03,
                 num_leaves=31, min_child_samples=30, feature_fraction=0.8,
                 bagging_fraction=0.85, bagging_freq=5, lambda_l2=1.0,
                 verbose=-1, seed=SEED)

    s2_A_oof = np.zeros((len(X_aug_tr), N_CLASSES))
    s2_A_test = np.zeros((len(X_aug_te), N_CLASSES))
    for tri, vai in folds:
        m = lgb.train(p_cls, lgb.Dataset(X_aug_tr.iloc[tri], y_tier.iloc[tri]),
                      2000, valid_sets=[lgb.Dataset(X_aug_tr.iloc[vai], y_tier.iloc[vai])],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        s2_A_oof[vai] = m.predict(X_aug_tr.iloc[vai], num_iteration=m.best_iteration)
        s2_A_test += m.predict(X_aug_te, num_iteration=m.best_iteration) / len(folds)
    print(f"  stage2_A acc={accuracy_score(y_tier, s2_A_oof.argmax(1)):.4f}")

    s2_B_oof = np.zeros(len(X_aug_tr))
    s2_B_test = np.zeros(len(X_aug_te))
    for tri, vai in folds:
        m = lgb.train(p_reg, lgb.Dataset(X_aug_tr.iloc[tri], y_rate.iloc[tri]),
                      2000, valid_sets=[lgb.Dataset(X_aug_tr.iloc[vai], y_rate.iloc[vai])],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        s2_B_oof[vai] = m.predict(X_aug_tr.iloc[vai], num_iteration=m.best_iteration)
        s2_B_test += m.predict(X_aug_te, num_iteration=m.best_iteration) / len(folds)
    print(f"  stage2_B  R²={r2_score(y_rate, s2_B_oof):.4f}")

    oof_A["stack2"] = s2_A_oof; test_A["stack2"] = s2_A_test
    oof_B["stack2"] = s2_B_oof; test_B["stack2"] = s2_B_test
    np.save(OOF_DIR / "stack2_A_oof.npy", s2_A_oof)
    np.save(OOF_DIR / "stack2_A_test.npy", s2_A_test)
    np.save(OOF_DIR / "stack2_B_oof.npy", s2_B_oof)
    np.save(OOF_DIR / "stack2_B_test.npy", s2_B_test)

    # --- Ensemble ---
    print_header("Ensemble (convex vs Ridge)")
    from stack import (optimise_blend_classification,
                       optimise_blend_regression)

    w_A = optimise_blend_classification(oof_A, y_tier.to_numpy())
    fA_conv_oof = sum(w_A[n] * oof_A[n] for n in w_A)
    fA_conv_test = sum(w_A[n] * test_A[n] for n in w_A)
    acc_conv = accuracy_score(y_tier, fA_conv_oof.argmax(1))
    _, rA_oof, rA_test = ridge_blend_classification(oof_A, y_tier.to_numpy(), test_A, alpha=1.0)
    acc_ridge = accuracy_score(y_tier, rA_oof.argmax(1))
    print(f"  A convex acc={acc_conv:.4f}   ridge acc={acc_ridge:.4f}")
    if acc_ridge > acc_conv:
        final_A_oof, final_A_test = rA_oof, rA_test
        acc_final, A_method = acc_ridge, "ridge"
    else:
        final_A_oof, final_A_test = fA_conv_oof, fA_conv_test
        acc_final, A_method = acc_conv, "convex"
    print(f"  A weights: { {k: round(float(v),3) for k,v in w_A.items()} }")

    w_B = optimise_blend_regression(oof_B, y_rate.to_numpy())
    fB_conv_oof = sum(w_B[n] * oof_B[n] for n in w_B)
    fB_conv_test = sum(w_B[n] * test_B[n] for n in w_B)
    r2_conv = r2_score(y_rate, fB_conv_oof)
    _, _, rB_oof, rB_test = ridge_blend_regression(oof_B, y_rate.to_numpy(), test_B, alpha=1.0)
    r2_ridge = r2_score(y_rate, rB_oof)
    print(f"  B convex R²={r2_conv:.4f}   ridge R²={r2_ridge:.4f}")
    if r2_ridge > r2_conv:
        final_B_oof, final_B_test = rB_oof, rB_test
        r2_final, B_method = r2_ridge, "ridge"
    else:
        final_B_oof, final_B_test = fB_conv_oof, fB_conv_test
        r2_final, B_method = r2_conv, "convex"
    print(f"  B weights: { {k: round(float(v),3) for k,v in w_B.items()} }")

    combined = 0.5 * acc_final + 0.5 * r2_final

    print_header("iter3c FINAL OOF estimate")
    print(f"  Accuracy       : {acc_final:.4f}  ({A_method})")
    print(f"  R²             : {r2_final:.4f}  ({B_method})")
    print(f"  Combined score : {combined:.4f}")
    print(f"  vs The Lions 0.88175 → gap = {combined - 0.88175:+.4f}")

    with open(OOF_DIR / "iter3_weights.json", "w") as f:
        json.dump({"A_method": A_method, "B_method": B_method,
                   "A_convex": {k: float(v) for k, v in w_A.items()},
                   "B_convex": {k: float(v) for k, v in w_B.items()},
                   "acc": acc_final, "r2": r2_final, "combined": combined,
                   "task_a_scores": {m: float(accuracy_score(y_tier, oof_A[m].argmax(1)))
                                     for m in oof_A},
                   "task_b_scores": {m: float(r2_score(y_rate, oof_B[m]))
                                     for m in oof_B}},
                  f, indent=2)

    tier_pred = final_A_test.argmax(1).astype(int)
    rate_pred = clip_rate(final_B_test)
    sub = pd.DataFrame({ID_COL: ids_test.astype(int).to_numpy(),
                        TARGET_A: tier_pred, TARGET_B: rate_pred})
    assert sub[ID_COL].is_unique and len(sub) == 15000
    assert set(sub[TARGET_A].unique()).issubset(set(range(N_CLASSES)))
    assert sub[TARGET_B].between(RATE_MIN, RATE_MAX).all()
    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"\n  wrote {OUT_DIR/'submission.csv'}  rows={len(sub)}")
    print(f"  tier_dist={sub[TARGET_A].value_counts().to_dict()}")
    print(f"  rate range=[{sub[TARGET_B].min():.2f}, {sub[TARGET_B].max():.2f}]")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
