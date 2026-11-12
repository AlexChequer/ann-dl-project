"""Stage 0: data for the regression deliverable — reuses the EDA pipeline.

Same split as the EDA (80/20, seed 42). The validation set is carved out of the
EDA training set; the preprocessor is fitted on the remaining training rows only.

The test set is only returned when asked for (`with_test=True`), which must
happen once, in Stage 5 (evaluate.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# The EDA pipeline is imported, not copied.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eda" / "code"))
from pipeline import (  # noqa: E402
    N_PRICE_BINS, SEED, build_preprocessor, load_split, transform_target,
)

VAL_SIZE = 0.2  # fraction of the EDA training set held out for validation

CODE_DIR = Path(__file__).resolve().parent
FIG_DIR = CODE_DIR.parent / "figures"


def load_data(with_test=False):
    """Return a dict with preprocessed features and targets.

    Keys: X_train, X_val (float32 arrays), y_train, y_val (price in US$, arrays),
    raw_train, raw_val (original feature frames, for subgroup analysis),
    feature_names and the fitted preprocessor `pre`.
    With `with_test=True`, also X_test, y_test and raw_test.
    """
    X_full, X_test, y_full, y_test = load_split()
    strata = pd.qcut(np.log1p(y_full), q=N_PRICE_BINS, labels=False, duplicates="drop")
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=VAL_SIZE, random_state=SEED, stratify=strata
    )
    pre = build_preprocessor().fit(X_train)  # fit on train only
    data = {
        "X_train": pre.transform(X_train).astype(np.float32),
        "X_val": pre.transform(X_val).astype(np.float32),
        "y_train": y_train.to_numpy(dtype=float),
        "y_val": y_val.to_numpy(dtype=float),
        "raw_train": X_train,
        "raw_val": X_val,
        "feature_names": list(pre.get_feature_names_out()),
        "pre": pre,
    }
    if with_test:
        data["X_test"] = pre.transform(X_test).astype(np.float32)
        data["y_test"] = y_test.to_numpy(dtype=float)
        data["raw_test"] = X_test
    return data


def to_log(price):
    """Model target: log1p(price), as decided in the EDA."""
    return np.asarray(transform_target(price), dtype=np.float32)


def to_price(log_pred):
    """Invert the target transform: back to US$."""
    return np.expm1(np.asarray(log_pred, dtype=float))


def regression_metrics(price_true, price_pred):
    """Metrics on the original scale (US$). MAE is the main metric."""
    return {
        "mae": float(mean_absolute_error(price_true, price_pred)),
        "rmse": float(root_mean_squared_error(price_true, price_pred)),
        "r2": float(r2_score(price_true, price_pred)),
        "rmse_log": float(root_mean_squared_error(
            np.log1p(price_true), np.log1p(np.clip(price_pred, 0, None))
        )),
    }


if __name__ == "__main__":
    d = load_data()
    print("train:", d["X_train"].shape, "val:", d["X_val"].shape)
    print("NaN train/val:", np.isnan(d["X_train"]).sum(), np.isnan(d["X_val"]).sum())
    print("median price train/val:", np.median(d["y_train"]), np.median(d["y_val"]))
