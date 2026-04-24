"""
predict.py
Take the final ensemble predictions from stack.run_stacking and write
Kaggle-formatted submission.csv. Also performs submission-level validation so
we fail loudly if anything violates the contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import (ID_COL, OUT_DIR, RATE_MAX, RATE_MIN, TARGET_A, TARGET_B,
                   clip_rate, print_header)

N_CLASSES = 5


def make_submission(ids: pd.Series, final_test_A: np.ndarray,
                    final_test_B: np.ndarray,
                    out_path: Path | None = None) -> Path:
    out_path = out_path or (OUT_DIR / "submission.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tier_pred = final_test_A.argmax(axis=1).astype(int)
    rate_pred = clip_rate(final_test_B)

    sub = pd.DataFrame({
        ID_COL: ids.astype(int).to_numpy(),
        TARGET_A: tier_pred,
        TARGET_B: rate_pred,
    })

    validate_submission(sub)
    sub.to_csv(out_path, index=False)
    print_header(f"Submission written → {out_path}")
    print(sub.head())
    print(f"  rows={len(sub)}  tier_dist={sub[TARGET_A].value_counts().to_dict()}")
    print(f"  rate range=[{sub[TARGET_B].min():.2f}, {sub[TARGET_B].max():.2f}]")
    return out_path


def validate_submission(sub: pd.DataFrame) -> None:
    assert list(sub.columns) == [ID_COL, TARGET_A, TARGET_B], \
        f"Bad columns: {list(sub.columns)}"
    assert sub[ID_COL].is_unique, "Duplicate Ids"
    assert sub[ID_COL].min() >= 0
    assert set(sub[TARGET_A].unique()).issubset(set(range(N_CLASSES))), \
        f"RiskTier outside 0..{N_CLASSES-1}"
    assert sub[TARGET_B].between(RATE_MIN, RATE_MAX).all(), "InterestRate out of range"
    # Round-to-2dp check
    assert np.allclose(sub[TARGET_B], sub[TARGET_B].round(2)), \
        "InterestRate must be rounded to 2 decimals"


if __name__ == "__main__":
    import json
    from utils import load_data
    from preprocessing import preprocess
    from features import engineer_features

    train, test = load_data()
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)
    _, _, _, _, ids_test = preprocess(train_fe, test_fe)

    final_A = np.load(OUT_DIR / "oof" / "final_A_test.npy")
    final_B = np.load(OUT_DIR / "oof" / "final_B_test.npy")
    make_submission(ids_test, final_A, final_B)
