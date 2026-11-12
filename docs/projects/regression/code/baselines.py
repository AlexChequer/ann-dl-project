"""Stage 1: baselines evaluated on the validation set.

Run from the repository root:  python docs/projects/regression/code/baselines.py
All baselines use the same pipeline and target as the network: they are fitted on
log1p(price) and their predictions are converted back to US$.
"""

import json

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from data import CODE_DIR, load_data, regression_metrics, to_log, to_price

OUT_JSON = CODE_DIR / "baselines_numbers.json"


def build_baselines():
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
    }


def fit_baselines(X_train, y_train):
    """Fit every baseline on log1p(price). Returns {name: fitted model}."""
    y_log = to_log(y_train)
    return {name: m.fit(X_train, y_log) for name, m in build_baselines().items()}


def main():
    d = load_data()
    models = fit_baselines(d["X_train"], d["y_train"])
    numbers = {
        "rows": {"train": len(d["y_train"]), "val": len(d["y_val"])},
        "val": {
            name: regression_metrics(d["y_val"], to_price(m.predict(d["X_val"])))
            for name, m in models.items()
        },
    }
    best = min(numbers["val"], key=lambda k: numbers["val"][k]["mae"])
    numbers["best_baseline"] = best
    OUT_JSON.write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
