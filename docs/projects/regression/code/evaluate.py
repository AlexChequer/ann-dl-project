"""Stage 5: the chosen configuration, retrained with 5 seeds, evaluated ONCE on the test set.

Run from the repository root:

    python docs/projects/regression/code/evaluate.py               # development: validation
    python docs/projects/regression/code/evaluate.py --split test  # the single test run

The default run uses validation only and writes nothing; it exists so the script can be
developed and debugged without touching the test set. `--split test` reads
best_config.json (decided in Stage 4), produces Figures 4 and 5, test_numbers.json and
test_predictions.csv (read by Stage 6).
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from baselines import fit_baselines  # noqa: E402
from data import CODE_DIR, FIG_DIR, load_data, regression_metrics, to_log, to_price  # noqa: E402
from model import count_parameters  # noqa: E402
from search import BEST_JSON, SEEDS, TARGET_GAIN, label, summarize  # noqa: E402
from train import BASE_CONFIG, predict_price, train_model  # noqa: E402

OUT_JSON = CODE_DIR / "test_numbers.json"
OUT_CSV = CODE_DIR / "test_predictions.csv"
CSV_SEED = 42
N_RESIDUAL_BINS = 5  # quintiles of the predicted price, for reading Figure 5


def eval_split(d, split):
    if split == "test":
        return d["X_test"], d["y_test"], d["raw_test"]
    return d["X_val"], d["y_val"], d["raw_val"]


def train_seeds(d, config, X_eval, y_eval):
    """Retrain the chosen configuration once per seed; early stopping on validation."""
    y_tr, y_va = to_log(d["y_train"]), to_log(d["y_val"])
    runs, preds, n_params = [], {}, None
    for seed in SEEDS:
        model, h = train_model(d["X_train"], y_tr, d["X_val"], y_va, config, seed)
        preds[seed] = predict_price(model, X_eval)
        n_params = count_parameters(model)
        runs.append({"seed": seed, **regression_metrics(y_eval, preds[seed]),
                     "best_epoch": h["best_epoch"], "stopped_epoch": h["stopped_epoch"]})
    return runs, preds, n_params


def criterion(mean, std, runs, baselines):
    """Stage 0 criterion and the gap to the best baseline against the seed variation."""
    best_name = min(baselines, key=lambda k: baselines[k]["mae"])
    base_mae = baselines[best_name]["mae"]
    gap = base_mae - mean["mae"]
    return {
        "best_baseline": best_name,
        "best_baseline_mae": base_mae,
        "threshold_mae": (1 - TARGET_GAIN) * base_mae,
        "gain_mean": 1 - mean["mae"] / base_mae,
        "gain_per_seed": {str(r["seed"]): 1 - r["mae"] / base_mae for r in runs},
        "seeds_meeting_criterion": sum(r["mae"] <= (1 - TARGET_GAIN) * base_mae for r in runs),
        "met": bool(mean["mae"] <= (1 - TARGET_GAIN) * base_mae),
        "gap_mae": gap,
        "gap_over_seed_std": gap / std["mae"],
        "worst_seed_mae": max(r["mae"] for r in runs),
        "all_seeds_beat_baseline": all(r["mae"] < base_mae for r in runs),
    }


def residual_summary(y_true, y_pred):
    """Bias of the predictions, overall and per quintile of the predicted price."""
    res = y_true - y_pred
    bins = pd.qcut(y_pred, q=N_RESIDUAL_BINS)
    per_bin = pd.DataFrame({"res": res, "bin": bins}).groupby("bin", observed=True)["res"]
    return {
        "mean_residual": float(res.mean()),
        "median_residual": float(np.median(res)),
        "share_underpredicted": float((res > 0).mean()),
        "mean_log_residual": float(np.mean(np.log1p(y_true) - np.log1p(y_pred))),
        "by_predicted_quintile": [
            {"pred_from": float(b.left), "pred_to": float(b.right),
             "median_residual": float(g.median()), "mean_residual": float(g.mean())}
            for b, g in per_bin
        ],
    }


def plot_pred_vs_actual(y_true, preds, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    lims = (8, 12000)
    for ax, (title, y_pred) in zip(axes, preds.items()):
        ax.scatter(y_true, y_pred, s=4, alpha=0.15, label="test listings")
        ax.plot(lims, lims, color="black", ls="--", label="perfect prediction (y = x)")
        ax.set(title=title, xscale="log", yscale="log", xlim=lims, ylim=lims,
               xlabel="actual price (US$, log scale)")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("predicted price (US$, log scale)")
    fig.suptitle("Predicted × actual price on the test set")
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=110)
    plt.close(fig)


def plot_residuals(y_true, y_pred, name):
    res = y_true - y_pred
    bins = pd.qcut(y_pred, q=N_RESIDUAL_BINS)
    grouped = pd.DataFrame({"p": y_pred, "r": res, "b": bins}).groupby("b", observed=True)
    med, avg = grouped.median(), grouped.mean()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(y_pred, res, s=4, alpha=0.15, label="test listings")
    ax.axhline(0, color="black", ls="--", label="zero residual")
    ax.plot(med["p"], med["r"], color="red", marker="o",
            label="median residual per predicted-price quintile")
    ax.plot(med["p"], avg["r"], color="darkorange", marker="s",
            label="mean residual per predicted-price quintile")
    ax.set(title="Residual (actual − predicted) × predicted price, test set, network seed 42",
           xscale="log", xlabel="predicted price (US$, log scale)",
           ylabel="residual (US$, symlog: linear in ±100)")
    ax.set_yscale("symlog", linthresh=100)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=110)
    plt.close(fig)


def write_predictions(raw_eval, y_eval, net_pred, base_preds):
    pd.DataFrame({
        "row_index": raw_eval.index,  # row of data/AB_NYC_2019.csv (0-based)
        "price_true": y_eval,
        "price_pred": net_pred,
        "pred_ridge": base_preds["ridge"],
        "pred_dummy": base_preds["dummy_mean"],
    }).to_csv(OUT_CSV, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["val", "test"], default="val")
    split = parser.parse_args().split

    best = json.loads(BEST_JSON.read_text())
    config = {**BASE_CONFIG, **best}
    d = load_data(with_test=(split == "test"))
    X_eval, y_eval, raw_eval = eval_split(d, split)

    models = fit_baselines(d["X_train"], d["y_train"])
    base_preds = {name: to_price(m.predict(X_eval)) for name, m in models.items()}
    baselines = {name: regression_metrics(y_eval, p) for name, p in base_preds.items()}
    runs, preds, n_params = train_seeds(d, best, X_eval, y_eval)
    mean, std = summarize(runs)

    numbers = {
        "split": split,
        "config": config,
        "label": label(best),
        "parameters": n_params,
        "seeds": list(SEEDS),
        "rows": len(y_eval),
        "baselines": baselines,
        "network": {"mean": mean, "std": std, "runs": runs},
        "criterion": criterion(mean, std, runs, baselines),
        "residuals_seed_42": residual_summary(y_eval, preds[CSV_SEED]),
    }
    print(json.dumps({k: numbers[k] for k in ("baselines", "criterion")}, indent=2))
    print("network mean:", json.dumps(mean), "\nnetwork std:", json.dumps(std))
    if split != "test":
        print("validation run: nothing written. Use --split test for the single test run.")
        return

    OUT_JSON.write_text(json.dumps(numbers, indent=2))
    write_predictions(raw_eval, y_eval, preds[CSV_SEED], base_preds)
    plot_pred_vs_actual(y_eval, {"MLP (seed 42)": preds[CSV_SEED],
                                 "Ridge (best baseline)": base_preds["ridge"]},
                        "fig04-pred-vs-actual.png")
    plot_residuals(y_eval, preds[CSV_SEED], "fig05-residuals.png")


if __name__ == "__main__":
    main()
