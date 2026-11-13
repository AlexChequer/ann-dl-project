"""Stage 6: error analysis — subgroups, worst cases and permutation importance.

Run from the repository root:  python docs/projects/regression/code/error_analysis.py

Two modes, chosen automatically from the files Stage 5 leaves in `code/`:

* **final** — `best_config.json` and `test_predictions.csv` both exist. The subgroup and
  worst-case analysis reads the **test** predictions written by Stage 5, so no new
  prediction is made on the test set here, and the permutation importance uses the chosen
  configuration retrained with seed 42.
* **provisional** — either file is missing. Everything runs on the **validation** set with
  the base network of Stage 3, so that the script can be written and checked before
  Stage 5 lands. Numbers from this mode do not go into the report.

The permutation importance always runs on the **validation** set, in both modes: the test
set is evaluated once, in Stage 5, and shuffling ten columns five times each is a repeated
probe, not a single evaluation.

Subgroups follow the EDA: `room_type` and `neighbourhood_group` (whose rare levels the EDA
counted), bands of the true price (quintiles plus the luxury tail above US$ 500) and the
`availability_365 == 0` flag. Every group is also measured against Ridge, the best
baseline, to show where the network's gain comes from.

Writes `code/error_numbers.json` and Figures 6-8.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from baselines import fit_baselines  # noqa: E402
from data import (  # noqa: E402
    CODE_DIR, FIG_DIR, SEED, load_data, regression_metrics, to_log, to_price,
)
from train import BASE_CONFIG, predict_price, train_model  # noqa: E402

OUT_JSON = CODE_DIR / "error_numbers.json"
BEST_CONFIG_JSON = CODE_DIR / "best_config.json"
TEST_PRED_CSV = CODE_DIR / "test_predictions.csv"
EDA_NUMBERS = CODE_DIR.parents[1] / "eda" / "code" / "eda_numbers.json"

FIG_START = 6        # Stage 5 ends at Figure 5
N_REPEATS = 5        # shuffles per column in the permutation importance
N_WORST = 10         # worst examples listed in the report
PRICE_BINS = 5       # quintiles of the true price
TAIL_PRICE = 500     # US$: the luxury tail the EDA flagged

FIG_SUBGROUP = f"fig{FIG_START:02d}-mae-by-subgroup.png"
FIG_BANDS = f"fig{FIG_START + 1:02d}-mae-by-price-band.png"
FIG_PERM = f"fig{FIG_START + 2:02d}-permutation-importance.png"

NET_COLOR = "#4C72B0"
RIDGE_COLOR = "#C44E52"


# --------------------------------------------------------------------------- inputs

def resolve_mode():
    """'final' once Stage 5 has written both of its files, 'provisional' before that."""
    if BEST_CONFIG_JSON.exists() and TEST_PRED_CSV.exists():
        return "final"
    return "provisional"


def read_test_predictions(index):
    """Stage 5 predictions, reindexed to the order of `raw_test`.

    The CSV carries the row number of `data/AB_NYC_2019.csv` (0-based), which is the index
    the pipeline keeps through `clean()` and the split.
    """
    pred = pd.read_csv(TEST_PRED_CSV)
    key = "row_index" if "row_index" in pred.columns else pred.columns[0]
    pred = pred.set_index(key)
    missing = index.difference(pred.index)
    extra = pred.index.difference(index)
    if len(missing) or len(extra):
        raise SystemExit(
            f"{TEST_PRED_CSV.name} does not cover the test set: {len(missing)} test rows "
            f"missing, {len(extra)} unknown rows present. Column used as index: '{key}'."
        )
    return pred.reindex(index)


def load_eda_rho():
    """Spearman |rho| with price per numerical column, as reported in the EDA."""
    try:
        eda = json.loads(EDA_NUMBERS.read_text())
        return eda["correlation"]["spearman"]["price"]
    except (OSError, KeyError, json.JSONDecodeError):
        return {}


# ------------------------------------------------------------------- group statistics

def build_frame(raw, y_true, y_pred, y_ridge=None):
    """Original features plus the true price, the prediction and the signed error."""
    df = raw.copy()
    df["price_true"] = np.asarray(y_true, dtype=float)
    df["price_pred"] = np.asarray(y_pred, dtype=float)
    df["error"] = df["price_pred"] - df["price_true"]
    df["abs_error"] = df["error"].abs()
    if y_ridge is not None:
        df["price_ridge"] = np.asarray(y_ridge, dtype=float)
        df["abs_error_ridge"] = (df["price_ridge"] - df["price_true"]).abs()
    return df


def stats(part, total):
    """MAE in US$ and the numbers needed to read it: size, price level and bias."""
    if len(part) == 0:
        return {"n": 0, "share_pct": 0.0, "mae": None, "bias": None,
                "mean_price": None, "mae_pct_of_price": None}
    mae = float(part["abs_error"].mean())
    mean_price = float(part["price_true"].mean())
    out = {
        "n": int(len(part)),
        "share_pct": round(100 * len(part) / total, 2),
        "mae": mae,
        "bias": float(part["error"].mean()),
        "mean_price": mean_price,
        "mae_pct_of_price": round(100 * mae / mean_price, 1),
    }
    if "abs_error_ridge" in part.columns:
        mae_ridge = float(part["abs_error_ridge"].mean())
        out["mae_ridge"] = mae_ridge
        out["gain_vs_ridge_pct"] = round(100 * (mae_ridge - mae) / mae_ridge, 1)
    return out


def subgroup_rows(df, column):
    """One row per level of a categorical column, sorted by MAE (largest first)."""
    rows = [{"group": str(level), **stats(part, len(df))}
            for level, part in df.groupby(column, observed=True)]
    return sorted(rows, key=lambda r: -(r["mae"] or 0))


def price_band_rows(df):
    """Quintiles of the true price, in order, plus the luxury tail as an extra row."""
    bands = pd.qcut(df["price_true"], q=PRICE_BINS, duplicates="drop")
    rows = []
    for i, (interval, part) in enumerate(df.groupby(bands, observed=True)):
        low = df["price_true"].min() if i == 0 else interval.left
        rows.append({"band": f"US$ {low:.0f}-{interval.right:.0f}", **stats(part, len(df))})
    tail = df[df["price_true"] > TAIL_PRICE]
    rows.append({"band": f"tail: > US$ {TAIL_PRICE}", **stats(tail, len(df))})
    return rows


def availability_rows(df):
    """The 'availability_365 == 0' flag the EDA found on a third of the listings."""
    zero = df["availability_365"] == 0
    return [
        {"group": "availability_365 == 0", **stats(df[zero], len(df))},
        {"group": "availability_365 > 0", **stats(df[~zero], len(df))},
    ]


def worst_rows(df, n):
    """The n largest absolute errors, with the original features."""
    cols = ["price_true", "price_pred", "abs_error", "room_type", "neighbourhood_group",
            "neighbourhood", "minimum_nights", "number_of_reviews", "reviews_per_month",
            "calculated_host_listings_count", "availability_365", "latitude", "longitude"]
    worst = df.sort_values("abs_error", ascending=False).head(n)
    out = []
    for row_index, row in worst[cols].iterrows():
        record = {"row_index": int(row_index)}
        for col in cols:
            value = row[col]
            record[col] = value if isinstance(value, str) else float(value)
        out.append(record)
    return out


def largest_error_group(subgroups, families, min_n=0):
    """The group with the largest MAE among the given families — summary row 8."""
    best = None
    for family in families:
        for row in subgroups[family]:
            label = row.get("group") or row.get("band")
            if row["mae"] is None or row["n"] < min_n:
                continue
            if best is None or row["mae"] > best["mae"]:
                best = {"family": family, "group": label, "mae": row["mae"],
                        "n": row["n"], "mean_price": row["mean_price"]}
    return best


# ----------------------------------------------------------- permutation importance

def permutation_importance(model, pre, raw, y_true, n_repeats, seed=SEED):
    """Increase in validation MAE when one ORIGINAL column is shuffled.

    The 10 columns of the raw frame are shuffled one at a time and pushed through the
    fitted preprocessor, so one shuffle moves every one-hot column that the feature
    produces (76 of the 91 model columns come from `neighbourhood` alone).
    """
    rng = np.random.default_rng(seed)
    base_mae = float(np.mean(np.abs(
        predict_price(model, pre.transform(raw).astype(np.float32)) - y_true
    )))
    eda_rho = load_eda_rho()
    features = []
    for column in raw.columns:
        values = raw[column].to_numpy()
        deltas = []
        for _ in range(n_repeats):
            shuffled = raw.copy()
            shuffled[column] = rng.permutation(values)
            mae = float(np.mean(np.abs(
                predict_price(model, pre.transform(shuffled).astype(np.float32)) - y_true
            )))
            deltas.append(mae - base_mae)
        entry = {
            "feature": column,
            "delta_mae_mean": float(np.mean(deltas)),
            "delta_mae_std": float(np.std(deltas)),
            "deltas": [float(d) for d in deltas],
        }
        if column in eda_rho:
            entry["eda_spearman_with_price"] = eda_rho[column]
        features.append(entry)
    features.sort(key=lambda f: -f["delta_mae_mean"])
    return {"split": "validation", "base_mae": base_mae, "n_repeats": n_repeats,
            "rows": int(len(raw)), "features": features}


# ------------------------------------------------------------------------- figures

def plot_subgroups(subgroups, overall_mae, split, path):
    families = [("room_type", "Room type"), ("neighbourhood_group", "Borough")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, title) in zip(axes, families):
        rows = sorted(subgroups[key], key=lambda r: r["mae"])
        y = np.arange(len(rows))
        ax.barh(y, [r["mae"] for r in rows], color=NET_COLOR, label="network")
        if rows and rows[0].get("mae_ridge") is not None:
            ax.plot([r["mae_ridge"] for r in rows], y, "D", color=RIDGE_COLOR,
                    markersize=6, linestyle="none", label="Ridge")
        ax.axvline(overall_mae, color="gray", ls="--",
                   label=f"all rows (US$ {overall_mae:.2f})")
        ax.set_yticks(y, [f'{r["group"]}\n(n = {r["n"]:,})' for r in rows], fontsize=8)
        ax.set(title=title, xlabel="MAE (US$)")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"MAE by subgroup on the {split} set — network against Ridge")
    fig.tight_layout()
    fig.savefig(FIG_DIR / path, dpi=110)
    plt.close(fig)


def plot_price_bands(rows, split, path):
    rows = [r for r in rows if r["n"] > 0]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(x, [r["mae"] for r in rows], color=NET_COLOR, label="MAE of the network (US$)")
    if rows[0].get("mae_ridge") is not None:
        ax.plot(x, [r["mae_ridge"] for r in rows], "D", color=RIDGE_COLOR,
                linestyle="none", markersize=6, label="MAE of Ridge (US$)")
    ax.set_xticks(x, [f'{r["band"]}\n(n = {r["n"]:,})' for r in rows], fontsize=8)
    ax.set(xlabel="band of the true price", ylabel="MAE (US$)")
    ax2 = ax.twinx()
    ax2.plot(x, [r["mae_pct_of_price"] for r in rows], "o-", color="#55A868",
             label="MAE as % of the band's mean price")
    ax2.set_ylabel("MAE / mean price of the band (%)")
    ax2.set_ylim(0, max(r["mae_pct_of_price"] for r in rows) * 1.25)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, fontsize=8, loc="upper left")
    ax.set_title(f"MAE by band of the true price — {split} set")
    fig.tight_layout()
    fig.savefig(FIG_DIR / path, dpi=110)
    plt.close(fig)


def plot_permutation(perm, path):
    features = perm["features"][::-1]  # ascending, so the largest ends up on top
    y = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(y, [f["delta_mae_mean"] for f in features],
            xerr=[f["delta_mae_std"] for f in features], color=NET_COLOR, capsize=3,
            label=f"mean of {perm['n_repeats']} shuffles (± 1 std)")
    ax.axvline(0, color="gray", lw=1)
    ax.set_yticks(y, [f["feature"] for f in features], fontsize=8)
    ax.set(xlabel="increase in validation MAE when the column is shuffled (US$)",
           ylabel="original column")
    ax.set_title("Permutation importance on the validation set\n"
                 f"(unshuffled MAE US$ {perm['base_mae']:.2f}, {perm['rows']:,} rows)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------- main

def main():
    mode = resolve_mode()
    config = dict(BASE_CONFIG)
    if mode == "final":
        config.update(json.loads(BEST_CONFIG_JSON.read_text()))
    print(f"[mode] {mode} — config: {config['hidden']}, dropout {config['dropout']}")

    d = load_data(with_test=(mode == "final"))

    # Stage 5's file is checked before anything is trained: a malformed CSV should fail in
    # a second, not after a full training run.
    pred = None
    if mode == "final":
        raw_eval, y_eval, split = d["raw_test"], d["y_test"], "test"
        pred = read_test_predictions(raw_eval.index)
        if not np.allclose(pred["price_true"].to_numpy(dtype=float), y_eval, atol=1e-6):
            raise SystemExit(
                "price_true in test_predictions.csv does not match the test target of the "
                "pipeline: Stage 5 and Stage 6 are not reading the same split."
            )
    else:
        raw_eval, y_eval, split = d["raw_val"], d["y_val"], "validation"

    model, history = train_model(d["X_train"], to_log(d["y_train"]),
                                 d["X_val"], to_log(d["y_val"]), config, seed=SEED)
    ridge = fit_baselines(d["X_train"], d["y_train"])["ridge"]

    if mode == "final":
        y_pred = pred["price_pred"].to_numpy(dtype=float)
        y_ridge = (pred["pred_ridge"].to_numpy(dtype=float)
                   if "pred_ridge" in pred.columns
                   else to_price(ridge.predict(d["X_test"])))
    else:
        y_pred = predict_price(model, d["X_val"])
        y_ridge = to_price(ridge.predict(d["X_val"]))

    df = build_frame(raw_eval, y_eval, y_pred, y_ridge)
    subgroups = {
        "room_type": subgroup_rows(df, "room_type"),
        "neighbourhood_group": subgroup_rows(df, "neighbourhood_group"),
        "price_band": price_band_rows(df),
        "availability_zero": availability_rows(df),
    }
    perm = permutation_importance(model, d["pre"], d["raw_val"], d["y_val"], N_REPEATS)

    numbers = {
        "mode": mode,
        "split": split,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in config.items()},
        "seed": SEED,
        "model_val_mae": float(np.mean(np.abs(predict_price(model, d["X_val"])
                                              - d["y_val"]))),
        "best_epoch": history["best_epoch"],
        "rows": int(len(df)),
        "overall": regression_metrics(y_eval, y_pred),
        "overall_ridge": regression_metrics(y_eval, y_ridge),
        "subgroups": subgroups,
        "largest_error_subgroup": largest_error_group(
            subgroups, ["room_type", "neighbourhood_group", "availability_zero"]),
        "largest_error_band": largest_error_group(subgroups, ["price_band"]),
        "worst_cases": worst_rows(df, N_WORST),
        "permutation": perm,
    }
    OUT_JSON.write_text(json.dumps(numbers, indent=2))

    plot_subgroups(subgroups, numbers["overall"]["mae"], split, FIG_SUBGROUP)
    plot_price_bands(subgroups["price_band"], split, FIG_BANDS)
    plot_permutation(perm, FIG_PERM)

    print(f"[{split}] MAE US$ {numbers['overall']['mae']:.2f} "
          f"(Ridge US$ {numbers['overall_ridge']['mae']:.2f}) on {numbers['rows']:,} rows")
    print("[worst subgroup]", numbers["largest_error_subgroup"])
    print("[top 3 permutation]",
          [(f["feature"], round(f["delta_mae_mean"], 2)) for f in perm["features"][:3]])
    print(f"[files] {OUT_JSON.name}, {FIG_SUBGROUP}, {FIG_BANDS}, {FIG_PERM}")


if __name__ == "__main__":
    main()
