"""
iter11.py — Add a tabular neural network as a base model.

Uses sklearn's MLPClassifier / MLPRegressor — simpler than building a
PyTorch transformer but still introduces a completely different error
geometry from gradient boosting. In an ensemble, NN residuals are largely
uncorrelated with boosting residuals, so the blend benefits even if the
NN alone is weaker than CatBoost.

We train the MLP on a standardised subset of the numeric feature matrix
(avoiding the OOF stacking features to keep it a "pure" base learner).
5-fold CV, same folds as everyone else. Augment the existing stage-2
feature set with MLP OOF, retrain stage-2, rebuild ensemble.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
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
from iter9 import multi_target_encode_v2, group_aggregate_features

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

N_CLASSES = 5


def fit_mlp_cls(X, y, Xt, folds, hidden=(256, 128), max_iter=80):
    """MLP classifier, standardised inputs, probability outputs."""
    oof = np.zeros((len(X), N_CLASSES))
    tp = np.zeros((len(Xt), N_CLASSES))
    for tri, vai in folds:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X.iloc[tri])
        X_va = scaler.transform(X.iloc[vai])
        X_te = scaler.transform(Xt)
        # Numerical safety: cap extreme values after scaling
        X_tr = np.clip(X_tr, -5, 5); X_va = np.clip(X_va, -5, 5); X_te = np.clip(X_te, -5, 5)
        m = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                          learning_rate_init=1e-3, batch_size=256,
                          alpha=1e-4, early_stopping=True,
                          validation_fraction=0.1, n_iter_no_change=10,
                          max_iter=max_iter, random_state=SEED, verbose=False)
        m.fit(X_tr, y.iloc[tri].to_numpy())
        oof[vai] = m.predict_proba(X_va)
        tp += m.predict_proba(X_te) / len(folds)
    return oof, tp


def fit_mlp_reg(X, y, Xt, folds, hidden=(256, 128), max_iter=80):
    oof = np.zeros(len(X))
    tp = np.zeros(len(Xt))
    for tri, vai in folds:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X.iloc[tri])
        X_va = scaler.transform(X.iloc[vai])
        X_te = scaler.transform(Xt)
        X_tr = np.clip(X_tr, -5, 5); X_va = np.clip(X_va, -5, 5); X_te = np.clip(X_te, -5, 5)
        m = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                         learning_rate_init=1e-3, batch_size=256,
                         alpha=1e-4, early_stopping=True,
                         validation_fraction=0.1, n_iter_no_change=10,
                         max_iter=max_iter, random_state=SEED, verbose=False)
        m.fit(X_tr, y.iloc[tri].to_numpy())
        oof[vai] = m.predict(X_va)
        tp += m.predict(X_te) / len(folds)
    return oof, tp


def main():
    set_seed(SEED)
    t0 = time.time()
    print_header("iter11 — Tabular NN as additional base learner")

    # ---- Load enriched feature set (same as iter9) ----
    train, test = load_data()
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)
    train_raw, test_raw = load_data()
    drop = [TARGET_A, TARGET_B, ID_COL]
    tr_raw_cats = train_raw.drop(columns=[c for c in drop if c in train_raw.columns]).copy()
    te_raw_cats = test_raw.drop(columns=[c for c in drop if c in test_raw.columns]).copy()
    for c in tr_raw_cats.columns:
        if tr_raw_cats[c].dtype == object:
            tr_raw_cats[c] = tr_raw_cats[c].fillna("NA").astype(str)
            te_raw_cats[c] = te_raw_cats[c].fillna("NA").astype(str)

    X_train, X_test, y_tier, y_rate, ids_test = preprocess(train_fe, test_fe)
    te_cols = [c for c in ["State", "JobCategory", "LoanPurpose",
                           "EmployerType", "EmploymentStatus"]
               if c in tr_raw_cats.columns]
    enc_tr, enc_te = multi_target_encode_v2(tr_raw_cats[te_cols], te_raw_cats[te_cols],
                                            te_cols, y_tier, y_rate, n_splits=10, smoothing=20)
    gagg_tr, gagg_te = group_aggregate_features(tr_raw_cats, te_raw_cats,
                                                y_tier, y_rate, n_splits=10, smoothing=20)
    old_te_cols = [c for c in X_train.columns
                   if c.endswith("_te_rate") or c.endswith("_te_p4")
                   or c.endswith("_te_std") or c.endswith("_te")]
    if old_te_cols:
        X_train = X_train.drop(columns=old_te_cols)
        X_test = X_test.drop(columns=old_te_cols)
    X_train_new = pd.concat([X_train.reset_index(drop=True),
                             enc_tr.reset_index(drop=True),
                             gagg_tr.reset_index(drop=True)], axis=1)
    X_test_new = pd.concat([X_test.reset_index(drop=True),
                            enc_te.reset_index(drop=True),
                            gagg_te.reset_index(drop=True)], axis=1)
    print(f"  enriched X_train={X_train_new.shape}")

    folds = get_folds(y_tier)

    # ---- Train MLP ----
    print_header("Train MLP classifier + regressor")
    t = time.time()
    mlp_A_oof, mlp_A_test = fit_mlp_cls(X_train_new, y_tier, X_test_new, folds)
    print(f"  mlp_A acc={accuracy_score(y_tier, mlp_A_oof.argmax(1)):.4f}  ({time.time()-t:.0f}s)")
    np.save(OOF_DIR / "v11_mlp_A_oof.npy", mlp_A_oof)
    np.save(OOF_DIR / "v11_mlp_A_test.npy", mlp_A_test)

    t = time.time()
    mlp_B_oof, mlp_B_test = fit_mlp_reg(X_train_new, y_rate, X_test_new, folds)
    print(f"  mlp_B R²={r2_score(y_rate, mlp_B_oof):.4f}  ({time.time()-t:.0f}s)")
    np.save(OOF_DIR / "v11_mlp_B_oof.npy", mlp_B_oof)
    np.save(OOF_DIR / "v11_mlp_B_test.npy", mlp_B_test)

    # ---- Load latest base OOFs, add MLP, rebuild stage-2 ----
    print_header("Stage-2 with MLP added")
    oof_A, test_A, oof_B, test_B = load_latest_oofs(OOF_DIR, y_tier, y_rate)

    # Prefer iter9 CAT and log-rate if present
    if (OOF_DIR / "v9_cat_A_oof.npy").exists():
        oof_A["cat"] = np.load(OOF_DIR / "v9_cat_A_oof.npy")
        test_A["cat"] = np.load(OOF_DIR / "v9_cat_A_test.npy")
    if (OOF_DIR / "v9_cat_B_oof.npy").exists():
        oof_B["cat"] = np.load(OOF_DIR / "v9_cat_B_oof.npy")
        test_B["cat"] = np.load(OOF_DIR / "v9_cat_B_test.npy")
    if (OOF_DIR / "v9_log_rate_B_oof.npy").exists():
        oof_B["log_rate"] = np.load(OOF_DIR / "v9_log_rate_B_oof.npy")
        test_B["log_rate"] = np.load(OOF_DIR / "v9_log_rate_B_test.npy")

    # Add MLP
    oof_A["mlp"] = mlp_A_oof; test_A["mlp"] = mlp_A_test
    oof_B["mlp"] = mlp_B_oof; test_B["mlp"] = mlp_B_test

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
    print_header("iter11 FINAL OOF")
    print(f"  Accuracy       : {acc_final:.4f}  ({A_method})")
    print(f"  R²             : {r2_final:.4f}  ({B_method})")
    print(f"  Combined score : {combined:.4f}")
    print(f"  vs The Lions 0.88175 → gap = {combined - 0.88175:+.4f}")
    print(f"  vs iter8 0.8407     → delta = {combined - 0.8407:+.4f}")

    with open(OOF_DIR / "iter11_weights.json", "w") as f:
        json.dump({"A_method": A_method, "B_method": B_method,
                   "acc": acc_final, "r2": r2_final, "combined": combined},
                  f, indent=2)

    # Determine best-so-far across all previous iters
    best = 0.8407
    for j in ["iter9_weights.json", "iter10_weights.json"]:
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
        print(f"  *** iter11 improves over {best:.4f} — submission updated ***")
    else:
        print(f"  iter11 ({combined:.4f}) did not beat {best:.4f}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
