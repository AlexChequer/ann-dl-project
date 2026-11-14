"""Stage 4: regularization comparison and hyperparameter search, on validation only.

Run from the repository root:  python docs/projects/regression/code/search.py
Produces Figure 3, search_numbers.json and best_config.json. Never loads the test set.

Every configuration is trained with the same 5 seeds; the ranking uses the mean
validation MAE in US$ over those seeds, so that one lucky seed cannot pick the winner.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from baselines import fit_baselines  # noqa: E402
from data import CODE_DIR, FIG_DIR, load_data, regression_metrics, to_log, to_price  # noqa: E402
from model import MLP, count_parameters  # noqa: E402
from train import BASE_CONFIG, predict_price, train_model  # noqa: E402

OUT_JSON = CODE_DIR / "search_numbers.json"
BEST_JSON = CODE_DIR / "best_config.json"
SEEDS = (42, 43, 44, 45, 46)
TARGET_GAIN = 0.05  # Stage 0 criterion: at least 5% below the best baseline

# (name, group, overrides of BASE_CONFIG). "base" has no regularization at all.
REGULARIZATION = [
    ("base", "base", {}),
    ("wd_0.01", "regularization", {"weight_decay": 1e-2}),
    ("wd_0.1", "regularization", {"weight_decay": 1e-1}),
    ("dropout_0.1", "regularization", {"dropout": 0.1}),
    ("dropout_0.3", "regularization", {"dropout": 0.3}),
    ("batchnorm", "regularization", {"batchnorm": True}),
]
SEARCH = [
    ("w64", "search", {"hidden": (64,)}),
    ("w256-128", "search", {"hidden": (256, 128)}),
    ("w256-128_do0.2", "search", {"hidden": (256, 128), "dropout": 0.2}),
    ("w256-128_do0.2_wd0.1", "search",
     {"hidden": (256, 128), "dropout": 0.2, "weight_decay": 1e-1}),
    ("w256-128_do0.2_lr3e-3", "search", {"hidden": (256, 128), "dropout": 0.2, "lr": 3e-3}),
    ("w256-128_do0.2_lr3e-4", "search", {"hidden": (256, 128), "dropout": 0.2, "lr": 3e-4}),
    ("w256-128-64_do0.2", "search", {"hidden": (256, 128, 64), "dropout": 0.2}),
    ("w256-128-64_bn_do0.1", "search",
     {"hidden": (256, 128, 64), "batchnorm": True, "dropout": 0.1}),
    ("w512-256-128_do0.3", "search", {"hidden": (512, 256, 128), "dropout": 0.3}),
]
# Round 2: the best of round 1 was the widest, most regularized network — at the edge
# of the grid — so the search continues in that direction (still validation only).
REFINEMENT = [
    ("w512-256-128_do0.4", "refinement", {"hidden": (512, 256, 128), "dropout": 0.4}),
    ("w512-256-128_do0.3_wd0.1", "refinement",
     {"hidden": (512, 256, 128), "dropout": 0.3, "weight_decay": 1e-1}),
    ("w512-256-128-64_do0.3", "refinement", {"hidden": (512, 256, 128, 64), "dropout": 0.3}),
    ("w1024-512-256_do0.3", "refinement", {"hidden": (1024, 512, 256), "dropout": 0.3}),
]


def add_availability_flag(X, raw):
    """EDA promise: append the binary flag availability_365 == 0 as a 92nd column."""
    flag = (raw["availability_365"].to_numpy() == 0).astype(np.float32)
    return np.column_stack([X, flag])


def label(overrides):
    """Readable description of a configuration, e.g. '256-128 · lr 1e-3 · dropout 0.2'."""
    cfg = {**BASE_CONFIG, **overrides}
    parts = ["-".join(str(u) for u in cfg["hidden"]), f"lr {cfg['lr']:g}"]
    if cfg["dropout"] > 0:
        parts.append(f"dropout {cfg['dropout']:g}")
    if cfg["weight_decay"] > 0:
        parts.append(f"wd {cfg['weight_decay']:g}")
    if cfg["batchnorm"]:
        parts.append("BatchNorm")
    return " · ".join(parts)


def summarize(runs):
    """Mean and sample std (ddof=1) over seeds of every metric in the runs."""
    keys = [k for k in runs[0] if k != "seed"]
    mean = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    std = {k: float(np.std([r[k] for r in runs], ddof=1)) for k in keys}
    return mean, std


def evaluate_config(X_tr, X_va, d, overrides):
    """Train one configuration with every seed; return per-seed validation results."""
    y_tr, y_va = to_log(d["y_train"]), to_log(d["y_val"])
    runs = []
    for seed in SEEDS:
        model, h = train_model(X_tr, y_tr, X_va, y_va, overrides, seed)
        metrics = regression_metrics(d["y_val"], predict_price(model, X_va))
        runs.append({"seed": seed, **metrics,
                     "best_epoch": h["best_epoch"], "stopped_epoch": h["stopped_epoch"]})
    return runs


def run_configs(specs, X_tr, X_va, d):
    results = []
    for name, group, overrides in specs:
        runs = evaluate_config(X_tr, X_va, d, overrides)
        mean, std = summarize(runs)
        cfg = {**BASE_CONFIG, **overrides}
        n_params = count_parameters(MLP(X_tr.shape[1], cfg["hidden"], cfg["dropout"],
                                        cfg["batchnorm"]))
        results.append({"name": name, "group": group, "label": label(overrides),
                        "overrides": overrides, "parameters": n_params,
                        "mean": mean, "std": std, "runs": runs})
        print(f"{name:26s} MAE {mean['mae']:.2f} ± {std['mae']:.2f}")
    return results


def flag_comparison(d, base_result):
    """Base network and Ridge with and without the availability flag."""
    X_tr = add_availability_flag(d["X_train"], d["raw_train"])
    X_va = add_availability_flag(d["X_val"], d["raw_val"])
    with_flag = run_configs([("base+avail_flag", "EDA flag", {})], X_tr, X_va, d)[0]
    ridge_plain = fit_baselines(d["X_train"], d["y_train"])["ridge"]
    ridge_flag = fit_baselines(X_tr, d["y_train"])["ridge"]
    ridge = {
        "without": regression_metrics(d["y_val"], to_price(ridge_plain.predict(d["X_val"]))),
        "with": regression_metrics(d["y_val"], to_price(ridge_flag.predict(X_va))),
    }
    return with_flag, {
        "share_flag_train": float(X_tr[:, -1].mean()),
        "network_mae_without": base_result["mean"]["mae"],
        "network_mae_with": with_flag["mean"]["mae"],
        "network_delta_mae": with_flag["mean"]["mae"] - base_result["mean"]["mae"],
        "network_std_without": base_result["std"]["mae"],
        "network_std_with": with_flag["std"]["mae"],
        "ridge": ridge,
        "ridge_delta_mae": ridge["with"]["mae"] - ridge["without"]["mae"],
    }


def changed_keys(overrides):
    """Only the keys whose value differs from BASE_CONFIG (tuples stored as lists)."""
    def norm(v):
        return list(v) if isinstance(v, tuple) else v
    return {k: norm(v) for k, v in overrides.items() if norm(v) != norm(BASE_CONFIG[k])}


def plot_search(results, ridge_mae, name):
    ordered = sorted(results, key=lambda r: r["mean"]["mae"], reverse=True)
    colors = {"base": "#555555", "regularization": "#1f77b4",
              "search": "#ff7f0e", "refinement": "#d62728", "EDA flag": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 6.5))
    # Points, not bars: the x axis does not start at 0, and bars would exaggerate gaps.
    for group, color in colors.items():
        idx = [i for i, r in enumerate(ordered) if r["group"] == group]
        ax.errorbar([ordered[i]["mean"]["mae"] for i in idx], idx,
                    xerr=[ordered[i]["std"]["mae"] for i in idx],
                    fmt="o", color=color, capsize=3, label=group)
    ax.set_yticks(np.arange(len(ordered)),
                  [r["label"] + (" + avail. flag" if r["group"] == "EDA flag" else "")
                   for r in ordered], fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(ridge_mae, color="black", ls="--", label=f"Ridge ({ridge_mae:.2f})")
    ax.axvline((1 - TARGET_GAIN) * ridge_mae, color="purple", ls=":",
               label=f"Ridge − 5% ({(1 - TARGET_GAIN) * ridge_mae:.2f})")
    lo = min(r["mean"]["mae"] - r["std"]["mae"] for r in results)
    ax.set_xlim(lo - 1.0, ridge_mae + 1.0)
    ax.set(title="Validation MAE per configuration (mean ± std over 5 seeds)",
           xlabel="validation MAE (US$), lower is better", ylabel="configuration")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=110)
    plt.close(fig)


def main():
    d = load_data()  # train and validation only
    results = run_configs(REGULARIZATION + SEARCH + REFINEMENT, d["X_train"], d["X_val"], d)
    base = next(r for r in results if r["name"] == "base")
    flag_result, flag = flag_comparison(d, base)

    ranked = sorted(results, key=lambda r: r["mean"]["mae"])  # flag kept apart
    best = ranked[0]
    ridge_mae = flag["ridge"]["without"]["mae"]
    numbers = {
        "seeds": list(SEEDS),
        "selection_rule": "lowest mean validation MAE (US$) over the seeds",
        "ridge_val_mae": ridge_mae,
        "ridge_val_mae_minus_5pct": (1 - TARGET_GAIN) * ridge_mae,
        "n_configs_ranked": len(ranked),
        "ranking": [r["name"] for r in ranked],
        "regularization_delta_vs_base": {
            r["name"]: r["mean"]["mae"] - base["mean"]["mae"]
            for r in results if r["group"] == "regularization"
        },
        "best": best["name"],
        "best_gain_vs_ridge_val": 1 - best["mean"]["mae"] / ridge_mae,
        "best_gain_vs_base_val": base["mean"]["mae"] - best["mean"]["mae"],
        "availability_flag": flag,
        "configs": ranked + [flag_result],
    }
    OUT_JSON.write_text(json.dumps(numbers, indent=2))
    BEST_JSON.write_text(json.dumps(changed_keys(best["overrides"]), indent=2))
    plot_search(ranked + [flag_result], ridge_mae, "fig03-search.png")
    print("best:", best["name"], json.dumps(changed_keys(best["overrides"])))


if __name__ == "__main__":
    main()
