"""
iter3.py — Iteration 3: push toward #1 on the leaderboard.

What's new vs run 2:
  - Reuses the 4 Task A base models from run 2 (on disk).
  - Adds a two-stage Task A classifier (is_tier4 + 4-class on non-tier-4).
  - Retrains all Task B models fresh with:
      * balanced HP (LR 0.03, num_leaves 95) — converges in reasonable time.
      * monotone constraints encoding underwriting domain knowledge.
      * tier-4 mixture-of-experts regressor.
      * DART boosting variant (diversity).
  - Multi-target encoding for high-card categoricals (mean rate + P(tier=4) + std).
  - Stage-2 stacking + convex weights + Ridge meta-blender.
  - Writes outputs/submission.csv.

Time budget on 2019-era 8-core Mac: ~60-75 min.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

from utils import (ID_COL, OOF_DIR, OUT_DIR, RATE_MAX, RATE_MIN, SEED,
                   TARGET_A, TARGET_B, clip_rate, get_folds, load_data,
                   print_header, set_seed)
from preprocessing import preprocess
from features import engineer_features
from advanced import (MONO_RATE, MONO_RISK, monotone_vector_for,
                      multi_target_encode, train_tier4_mixture,
                      train_two_stage_tier, train_dart_cls, train_dart_reg,
                      ridge_blend_classification, ridge_blend_regression)

N_CLASSES = 5


def load_raw():
    train, test = load_data()
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)
    return train_fe, test_fe


def add_multi_te(X_train, X_test, train_raw, test_raw, y_tier, y_rate):
    """
    Enrich feature set with multi-target encodings for the high-cardinality
    categoricals identified in EDA. Uses the raw string columns from the
    original frames since preprocess() has already integer-encoded them.
    """
    candidates = ["State", "JobCategory", "LoanPurpose",
                  "EmployerType", "EmploymentStatus"]
    cats_present = [c for c in candidates if c in train_raw.columns]
    if not cats_present:
        return X_train, X_test
    tr_raw = train_raw[cats_present].fillna("NA").astype(str)
    te_raw = test_raw[cats_present].fillna("NA").astype(str)
    enc_tr, enc_te = multi_target_encode(
        tr_raw, te_raw, cats_present, y_tier, y_rate)
    return (pd.concat([X_train.reset_index(drop=True),
                       enc_tr.reset_index(drop=True)], axis=1),
            pd.concat([X_test.reset_index(drop=True),
                       enc_te.reset_index(drop=True)], axis=1))


# ---------------------------------------------------------------------------
# Task B boosters with monotone constraints
# ---------------------------------------------------------------------------

def fit_lgb_reg_mono(X, y, Xt, folds, rounds=3500):
    mono = monotone_vector_for(X.columns, MONO_RATE)
    # NB: LightGBM rejects monotone_constraints + regression_l1. Use L2 here
    # (squared-error) which is compatible; we keep a separate non-mono L1
    # model below so the ensemble still sees an L1 learner.
    p = dict(objective="regression", metric="rmse",
             learning_rate=0.03, num_leaves=95, min_child_samples=20,
             feature_fraction=0.80, bagging_fraction=0.85, bagging_freq=5,
             lambda_l1=0.1, lambda_l2=1.0,
             monotone_constraints=mono, monotone_constraints_method="basic",
             verbose=-1, seed=SEED)
    oof = np.zeros(len(X)); tp = np.zeros(len(Xt))
    for tri, vai in folds:
        d_tr = lgb.Dataset(X.iloc[tri], y.iloc[tri])
        d_va = lgb.Dataset(X.iloc[vai], y.iloc[vai])
        m = lgb.train(p, d_tr, rounds, valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        oof[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration)
        tp += m.predict(Xt, num_iteration=m.best_iteration) / len(folds)
    return oof, tp


def fit_xgb_reg_mono(X, y, Xt, folds, rounds=3500):
    # XGBoost requires the tuple-as-string form for monotone_constraints
    mono_vec = monotone_vector_for(X.columns, MONO_RATE)
    mono_str = "(" + ",".join(str(int(x)) for x in mono_vec) + ")"
    p = dict(objective="reg:squarederror", eval_metric="rmse",
             learning_rate=0.03, max_depth=8, subsample=0.85,
             colsample_bytree=0.70, min_child_weight=3, reg_alpha=0.1,
             reg_lambda=1.0, gamma=0.01, tree_method="hist",
             monotone_constraints=mono_str,
             random_state=SEED, verbosity=0)
    oof = np.zeros(len(X)); tp = np.zeros(len(Xt))
    dt = xgb.DMatrix(Xt)
    for tri, vai in folds:
        d_tr = xgb.DMatrix(X.iloc[tri], label=y.iloc[tri])
        d_va = xgb.DMatrix(X.iloc[vai], label=y.iloc[vai])
        m = xgb.train(p, d_tr, rounds, [(d_va, "v")],
                      early_stopping_rounds=150, verbose_eval=False)
        oof[vai] = m.predict(d_va, iteration_range=(0, m.best_iteration + 1))
        tp += m.predict(dt, iteration_range=(0, m.best_iteration + 1)) / len(folds)
    return oof, tp


def fit_cat_reg(X, y, Xt, folds, iters=3000):
    oof = np.zeros(len(X)); tp = np.zeros(len(Xt))
    for tri, vai in folds:
        m = CatBoostRegressor(loss_function="RMSE", iterations=iters,
                              learning_rate=0.04, depth=8, l2_leaf_reg=5.0,
                              random_strength=0.5, bagging_temperature=0.2,
                              border_count=254, random_seed=SEED,
                              verbose=False, allow_writing_files=False,
                              early_stopping_rounds=150)
        m.fit(X.iloc[tri], y.iloc[tri], eval_set=(X.iloc[vai], y.iloc[vai]),
              use_best_model=True)
        oof[vai] = m.predict(X.iloc[vai])
        tp += m.predict(Xt) / len(folds)
    return oof, tp


# ---------------------------------------------------------------------------
# Task A: two-stage classifier added to the existing 4
# ---------------------------------------------------------------------------

def fit_lgb_cls_mono(X, y, Xt, folds, rounds=3500):
    mono = monotone_vector_for(X.columns, MONO_RISK)
    p = dict(objective="multiclass", num_class=N_CLASSES, metric="multi_logloss",
             learning_rate=0.03, num_leaves=95, min_child_samples=20,
             feature_fraction=0.80, bagging_fraction=0.85, bagging_freq=5,
             lambda_l1=0.1, lambda_l2=1.0,
             monotone_constraints=mono, monotone_constraints_method="advanced",
             verbose=-1, seed=SEED)
    oof = np.zeros((len(X), N_CLASSES)); tp = np.zeros((len(Xt), N_CLASSES))
    for tri, vai in folds:
        d_tr = lgb.Dataset(X.iloc[tri], y.iloc[tri])
        d_va = lgb.Dataset(X.iloc[vai], y.iloc[vai])
        m = lgb.train(p, d_tr, rounds, valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        oof[vai] = m.predict(X.iloc[vai], num_iteration=m.best_iteration)
        tp += m.predict(Xt, num_iteration=m.best_iteration) / len(folds)
    return oof, tp


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    t0 = time.time()
    print_header("Iter 3 — Load + engineer + enrich preprocessing")
    train_fe, test_fe = load_raw()
    train_raw, test_raw = load_data()  # raw strings for multi-TE
    X_train, X_test, y_tier, y_rate, ids_test = preprocess(train_fe, test_fe)
    X_train, X_test = add_multi_te(X_train, X_test, train_raw, test_raw,
                                   y_tier, y_rate)
    print(f"  X_train={X_train.shape}  X_test={X_test.shape}")

    folds = get_folds(y_tier)

    # ---- Task A ----
    print_header("Task A — retrain LGB_mono + add two-stage model")
    oof_A, test_A = {}, {}

    # Reuse existing run-2 OOFs
    for m in ["lgb", "xgb", "cat", "lgb_ord"]:
        oof_A[m] = np.load(OOF_DIR / f"{m}_A_oof.npy")
        test_A[m] = np.load(OOF_DIR / f"{m}_A_test.npy")
        acc = accuracy_score(y_tier, oof_A[m].argmax(1))
        print(f"  [reused] {m}_A acc={acc:.4f}")

    t = time.time()
    oof_A["lgb_mono"], test_A["lgb_mono"] = fit_lgb_cls_mono(
        X_train, y_tier, X_test, folds)
    np.save(OOF_DIR / "lgb_mono_A_oof.npy", oof_A["lgb_mono"])
    np.save(OOF_DIR / "lgb_mono_A_test.npy", test_A["lgb_mono"])
    print(f"  lgb_mono_A acc={accuracy_score(y_tier, oof_A['lgb_mono'].argmax(1)):.4f}  ({time.time()-t:.0f}s)")

    t = time.time()
    oof_A["two_stage"], test_A["two_stage"] = train_two_stage_tier(
        X_train, y_tier, X_test, folds, rounds=3500)
    np.save(OOF_DIR / "two_stage_A_oof.npy", oof_A["two_stage"])
    np.save(OOF_DIR / "two_stage_A_test.npy", test_A["two_stage"])
    print(f"  two_stage_A acc={accuracy_score(y_tier, oof_A['two_stage'].argmax(1)):.4f}  ({time.time()-t:.0f}s)")

    # ---- Task B ----
    print_header("Task B — fresh training with monotone + mixture + DART")
    oof_B, test_B = {}, {}

    t = time.time()
    oof_B["lgb"], test_B["lgb"] = fit_lgb_reg_mono(X_train, y_rate, X_test, folds)
    np.save(OOF_DIR / "lgb_B_oof.npy", oof_B["lgb"])
    np.save(OOF_DIR / "lgb_B_test.npy", test_B["lgb"])
    print(f"  lgb_B  R²={r2_score(y_rate, oof_B['lgb']):.4f}  ({time.time()-t:.0f}s)")

    t = time.time()
    oof_B["xgb"], test_B["xgb"] = fit_xgb_reg_mono(X_train, y_rate, X_test, folds)
    np.save(OOF_DIR / "xgb_B_oof.npy", oof_B["xgb"])
    np.save(OOF_DIR / "xgb_B_test.npy", test_B["xgb"])
    print(f"  xgb_B  R²={r2_score(y_rate, oof_B['xgb']):.4f}  ({time.time()-t:.0f}s)")

    t = time.time()
    oof_B["cat"], test_B["cat"] = fit_cat_reg(X_train, y_rate, X_test, folds)
    np.save(OOF_DIR / "cat_B_oof.npy", oof_B["cat"])
    np.save(OOF_DIR / "cat_B_test.npy", test_B["cat"])
    print(f"  cat_B  R²={r2_score(y_rate, oof_B['cat']):.4f}  ({time.time()-t:.0f}s)")

    t = time.time()
    oof_B["mix"], test_B["mix"] = train_tier4_mixture(
        X_train, y_rate, y_tier, X_test, folds, rounds=3000)
    np.save(OOF_DIR / "mix_B_oof.npy", oof_B["mix"])
    np.save(OOF_DIR / "mix_B_test.npy", test_B["mix"])
    print(f"  mix_B  R²={r2_score(y_rate, oof_B['mix']):.4f}  ({time.time()-t:.0f}s)")

    t = time.time()
    oof_B["dart"], test_B["dart"] = train_dart_reg(X_train, y_rate, X_test, folds,
                                                   rounds=2500)
    np.save(OOF_DIR / "dart_B_oof.npy", oof_B["dart"])
    np.save(OOF_DIR / "dart_B_test.npy", test_B["dart"])
    print(f"  dart_B R²={r2_score(y_rate, oof_B['dart']):.4f}  ({time.time()-t:.0f}s)")

    # ---- Stage 2 stacking ----
    print_header("Stage-2 — augmented-feature stacking")
    # Concatenate original X with all OOFs
    parts_tr = [X_train.reset_index(drop=True)]
    parts_te = [X_test.reset_index(drop=True)]
    for m, arr in oof_A.items():
        cols = [f"oofA_{m}_p{k}" for k in range(arr.shape[1])]
        parts_tr.append(pd.DataFrame(arr, columns=cols))
        parts_te.append(pd.DataFrame(test_A[m], columns=cols))
    # Include the ordinal float prediction
    try:
        ord_f = np.load(OOF_DIR / "lgb_ord_A_oof_float.npy")
        ord_t = np.load(OOF_DIR / "lgb_ord_A_test_float.npy")
        parts_tr.append(pd.DataFrame({"oofA_ord_float": ord_f}))
        parts_te.append(pd.DataFrame({"oofA_ord_float": ord_t}))
    except Exception:
        pass
    for m, arr in oof_B.items():
        parts_tr.append(pd.DataFrame({f"oofB_{m}": arr}))
        parts_te.append(pd.DataFrame({f"oofB_{m}": test_B[m]}))
    X_aug_tr = pd.concat(parts_tr, axis=1)
    X_aug_te = pd.concat(parts_te, axis=1)

    # Stage-2 LGBM per task
    p_s2_cls = dict(objective="multiclass", num_class=N_CLASSES,
                    metric="multi_logloss", learning_rate=0.03, num_leaves=31,
                    min_child_samples=30, feature_fraction=0.8,
                    bagging_fraction=0.85, bagging_freq=5, lambda_l2=1.0,
                    verbose=-1, seed=SEED)
    p_s2_reg = dict(objective="regression", metric="rmse", learning_rate=0.03,
                    num_leaves=31, min_child_samples=30, feature_fraction=0.8,
                    bagging_fraction=0.85, bagging_freq=5, lambda_l2=1.0,
                    verbose=-1, seed=SEED)

    s2_A_oof = np.zeros((len(X_aug_tr), N_CLASSES))
    s2_A_test = np.zeros((len(X_aug_te), N_CLASSES))
    for tri, vai in folds:
        m = lgb.train(p_s2_cls, lgb.Dataset(X_aug_tr.iloc[tri], y_tier.iloc[tri]),
                      2000, valid_sets=[lgb.Dataset(X_aug_tr.iloc[vai], y_tier.iloc[vai])],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        s2_A_oof[vai] = m.predict(X_aug_tr.iloc[vai], num_iteration=m.best_iteration)
        s2_A_test += m.predict(X_aug_te, num_iteration=m.best_iteration) / len(folds)

    s2_B_oof = np.zeros(len(X_aug_tr))
    s2_B_test = np.zeros(len(X_aug_te))
    for tri, vai in folds:
        m = lgb.train(p_s2_reg, lgb.Dataset(X_aug_tr.iloc[tri], y_rate.iloc[tri]),
                      2000, valid_sets=[lgb.Dataset(X_aug_tr.iloc[vai], y_rate.iloc[vai])],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        s2_B_oof[vai] = m.predict(X_aug_tr.iloc[vai], num_iteration=m.best_iteration)
        s2_B_test += m.predict(X_aug_te, num_iteration=m.best_iteration) / len(folds)

    oof_A["stack2"] = s2_A_oof; test_A["stack2"] = s2_A_test
    oof_B["stack2"] = s2_B_oof; test_B["stack2"] = s2_B_test
    np.save(OOF_DIR / "stack2_A_oof.npy", s2_A_oof)
    np.save(OOF_DIR / "stack2_A_test.npy", s2_A_test)
    np.save(OOF_DIR / "stack2_B_oof.npy", s2_B_oof)
    np.save(OOF_DIR / "stack2_B_test.npy", s2_B_test)

    print(f"  stage2_A acc={accuracy_score(y_tier, s2_A_oof.argmax(1)):.4f}")
    print(f"  stage2_B  R²={r2_score(y_rate, s2_B_oof):.4f}")

    # ---- Ensemble: convex Nelder-Mead + Ridge meta-blender, pick winner ----
    print_header("Ensemble selection")

    # --- Task A: convex Nelder-Mead (from stack.py) + Ridge multi-blend ---
    from stack import optimise_blend_classification
    w_A = optimise_blend_classification(oof_A, y_tier.to_numpy())
    final_A_conv_oof = sum(w_A[n] * oof_A[n] for n in w_A)
    final_A_conv_test = sum(w_A[n] * test_A[n] for n in w_A)
    acc_conv = accuracy_score(y_tier, final_A_conv_oof.argmax(1))
    print(f"  A convex  acc={acc_conv:.4f}  weights={ {k: round(float(v),3) for k,v in w_A.items()} }")

    # Ridge multi-class blender
    ridge_model_A, ridge_A_blend_oof, ridge_A_blend_test = ridge_blend_classification(
        oof_A, y_tier.to_numpy(), test_A, alpha=1.0)
    acc_ridge = accuracy_score(y_tier, ridge_A_blend_oof.argmax(1))
    print(f"  A ridge   acc={acc_ridge:.4f}")

    if acc_ridge > acc_conv:
        final_A_oof = ridge_A_blend_oof
        final_A_test = ridge_A_blend_test
        acc_final = acc_ridge; A_method = "ridge"
    else:
        final_A_oof = final_A_conv_oof
        final_A_test = final_A_conv_test
        acc_final = acc_conv; A_method = "convex"

    # --- Task B: convex Nelder-Mead + Ridge scalar ---
    from stack import optimise_blend_regression
    w_B = optimise_blend_regression(oof_B, y_rate.to_numpy())
    final_B_conv_oof = sum(w_B[n] * oof_B[n] for n in w_B)
    final_B_conv_test = sum(w_B[n] * test_B[n] for n in w_B)
    r2_conv = r2_score(y_rate, final_B_conv_oof)
    print(f"  B convex  R²={r2_conv:.4f}  weights={ {k: round(float(v),3) for k,v in w_B.items()} }")

    ridge_coefs_B, ridge_intercept_B, ridge_B_oof, ridge_B_test = ridge_blend_regression(
        oof_B, y_rate.to_numpy(), test_B, alpha=1.0)
    r2_ridge_B = r2_score(y_rate, ridge_B_oof)
    print(f"  B ridge   R²={r2_ridge_B:.4f}  coefs={ {k: round(float(v),3) for k,v in ridge_coefs_B.items()} } intercept={ridge_intercept_B:.3f}")

    if r2_ridge_B > r2_conv:
        final_B_oof = ridge_B_oof; final_B_test = ridge_B_test
        r2_final = r2_ridge_B; B_method = "ridge"
    else:
        final_B_oof = final_B_conv_oof; final_B_test = final_B_conv_test
        r2_final = r2_conv; B_method = "convex"

    combined = 0.5 * acc_final + 0.5 * r2_final

    print_header("Iter 3 final OOF estimate of leaderboard")
    print(f"  Accuracy       : {acc_final:.4f}  ({A_method})")
    print(f"  R²             : {r2_final:.4f}  ({B_method})")
    print(f"  Combined score : {combined:.4f}")

    with open(OOF_DIR / "iter3_weights.json", "w") as f:
        json.dump({"A_method": A_method, "B_method": B_method,
                   "A_convex": {k: float(v) for k, v in w_A.items()},
                   "B_convex": {k: float(v) for k, v in w_B.items()},
                   "acc": acc_final, "r2": r2_final,
                   "combined": combined}, f, indent=2)

    # ---- Submission ----
    print_header("Submission")
    tier_pred = final_A_oof.argmax(1) if False else final_A_test.argmax(1)
    # (First expression was a readability mistake — always use TEST predictions)
    tier_pred = final_A_test.argmax(1).astype(int)
    rate_pred = clip_rate(final_B_test)

    sub = pd.DataFrame({ID_COL: ids_test.astype(int).to_numpy(),
                        TARGET_A: tier_pred, TARGET_B: rate_pred})
    assert sub[ID_COL].is_unique and len(sub) == 15000
    assert set(sub[TARGET_A].unique()).issubset(set(range(N_CLASSES)))
    assert sub[TARGET_B].between(RATE_MIN, RATE_MAX).all()

    out_path = OUT_DIR / "submission.csv"
    sub.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")
    print(f"  rows={len(sub)}  tier_dist={sub[TARGET_A].value_counts().to_dict()}")
    print(f"  rate range=[{sub[TARGET_B].min():.2f}, {sub[TARGET_B].max():.2f}]")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
