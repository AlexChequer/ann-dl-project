"""Stages 1 to 3: report numbers and Figures 1 to 7.

Run from the repository root:  python docs/projects/eda/code/eda_figures.py
Every distribution analysis uses the TRAINING set only.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from pipeline import (  # noqa: E402
    CAT_COLS, MAX_MINIMUM_NIGHTS, NUM_COLS, TARGET, clean, load_raw, split,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_JSON = Path(__file__).resolve().parent / "eda_numbers.json"
TOP_NEIGHBOURHOODS = 20
SCATTER_SAMPLE = 5000
SEED = 42


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=110)
    plt.close(fig)


def stage1_quality(raw):
    missing = raw.isna().sum()
    return {
        "shape_raw": list(raw.shape),
        "missing_count": missing[missing > 0].to_dict(),
        "missing_pct": (100 * missing[missing > 0] / len(raw)).round(2).to_dict(),
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_ids": int(raw["id"].duplicated().sum()),
        "price_zero": int((raw[TARGET] == 0).sum()),
        "min_nights_gt_365": int((raw["minimum_nights"] > MAX_MINIMUM_NIGHTS).sum()),
        "availability_zero": int((raw["availability_365"] == 0).sum()),
        "nan_rpm_iff_zero_reviews": bool(
            ((raw["number_of_reviews"] == 0) == raw["reviews_per_month"].isna()).all()
        ),
        "last_review_range": [raw["last_review"].min(), raw["last_review"].max()],
    }


def stage1_target(y_train):
    log_y = np.log1p(y_train)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(y_train, bins=100, color="#4C72B0", label="price")
    axes[0].set(title="price (US$/night)", xlabel="price (US$)", ylabel="frequency")
    axes[1].hist(log_y, bins=60, color="#DD8452", label="log1p(price)")
    axes[1].set(title="log1p(price)", xlabel="log1p(price)", ylabel="frequency")
    for ax in axes:
        ax.legend()
    fig.suptitle("Figure 1 — Target distribution (train)")
    save(fig, "fig01-target.png")
    return {
        "mean": round(y_train.mean(), 2), "median": float(y_train.median()),
        "std": round(y_train.std(), 2), "min": int(y_train.min()),
        "max": int(y_train.max()), "skew": round(y_train.skew(), 2),
        "skew_log": round(log_y.skew(), 2),
        "pct_gt_500": round(100 * (y_train > 500).mean(), 2),
    }


def stage2_numeric(X_train):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, col in zip(axes.flat, NUM_COLS):
        ax.hist(X_train[col].dropna(), bins=60, color="#4C72B0", label=col)
        ax.set(title=col, xlabel=col, ylabel="frequency")
        if X_train[col].skew() > 3:
            ax.set_yscale("log")
            ax.set_ylabel("frequency (log)")
        ax.legend(fontsize=7)
    axes.flat[-1].axis("off")
    fig.suptitle("Figure 2 — Histograms of the numerical features (train)")
    save(fig, "fig02-numeric-hist.png")
    desc = X_train[NUM_COLS].describe().T
    desc["skew"] = X_train[NUM_COLS].skew()
    return desc.round(3).to_dict(orient="index")


def stage2_categorical(X_train):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                             gridspec_kw={"width_ratios": [1, 1, 2]})
    for ax, col in zip(axes[:2], ["neighbourhood_group", "room_type"]):
        counts = X_train[col].value_counts()
        ax.bar(counts.index, counts.values, color="#55A868", label="listings")
        ax.set(title=col, xlabel=col, ylabel="number of listings")
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
    top = X_train["neighbourhood"].value_counts().head(TOP_NEIGHBOURHOODS)
    axes[2].barh(top.index[::-1], top.values[::-1], color="#8172B3", label="listings")
    axes[2].set(title=f"neighbourhood (top {TOP_NEIGHBOURHOODS} of "
                f"{X_train['neighbourhood'].nunique()})",
                xlabel="number of listings", ylabel="neighbourhood")
    axes[2].legend()
    fig.suptitle("Figure 3 — Categorical frequencies (train)")
    save(fig, "fig03-categorical-bars.png")
    nb = X_train["neighbourhood"].value_counts()
    return {
        "cardinality": {c: int(X_train[c].nunique()) for c in CAT_COLS},
        "freq_pct": {c: (100 * X_train[c].value_counts(normalize=True)).round(2).to_dict()
                     for c in ["neighbourhood_group", "room_type"]},
        "neighbourhood_lt50": int((nb < 50).sum()),
        "neighbourhood_lt50_rows_pct": round(100 * nb[nb < 50].sum() / len(X_train), 2),
        "neighbourhood_singletons": int((nb == 1).sum()),
        "top5_neighbourhood": nb.head(5).to_dict(),
    }


def stage3_correlation(train):
    cols = NUM_COLS + [TARGET]
    corr = train[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax,
                cbar_kws={"label": "Spearman ρ"})
    ax.set(title="Figure 4 — Spearman correlation (train)", xlabel="feature",
           ylabel="feature")
    save(fig, "fig04-spearman.png")

    sample = train.sample(SCATTER_SAMPLE, random_state=SEED)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(sample["number_of_reviews"], sample["reviews_per_month"], s=5,
                    alpha=0.4, label=f"sample of {SCATTER_SAMPLE}")
    axes[0].set(xscale="symlog", yscale="symlog", xlabel="number_of_reviews",
                ylabel="reviews_per_month", title="reviews: total × per month")
    axes[1].scatter(sample["longitude"], sample["latitude"], s=5, alpha=0.5,
                    c=np.log1p(sample[TARGET]), cmap="viridis",
                    label="color = log1p(price)")
    axes[1].set(xlabel="longitude", ylabel="latitude", title="location × price")
    for ax in axes:
        ax.legend(loc="upper left")
    fig.suptitle("Figure 5 — Scatter of numerical pairs (train)")
    save(fig, "fig05-scatter.png")

    pairs = corr.drop(index=TARGET, columns=TARGET).abs().where(
        np.triu(np.ones((len(NUM_COLS),) * 2, dtype=bool), k=1)).stack()
    return {
        "spearman": corr.round(3).to_dict(),
        "top_pairs": {f"{a} × {b}": round(float(corr.loc[a, b]), 3)
                      for a, b in pairs.sort_values(ascending=False).head(3).index},
        "pearson_reviews": round(train["number_of_reviews"].corr(
            train["reviews_per_month"]), 3),
    }


def stage3_cat_target(train):
    log_price = np.log1p(train[TARGET]).rename("log1p(price)")
    data = train.assign(**{"log1p(price)": log_price})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axes, ["room_type", "neighbourhood_group"]):
        order = data.groupby(col)["log1p(price)"].median().sort_values().index
        sns.boxplot(data=data, x=col, y="log1p(price)", order=order, ax=ax,
                    hue=col, legend=True, showfliers=False)
        ax.set(title=f"log1p(price) by {col}", xlabel=col, ylabel="log1p(price)")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Figure 6 — Target distribution by category (train)")
    save(fig, "fig06-target-by-category.png")
    stats = {}
    for col in ["room_type", "neighbourhood_group"]:
        g = train.groupby(col)[TARGET]
        stats[col] = pd.DataFrame({"median": g.median(), "mean": g.mean().round(1),
                                   "iqr": (g.quantile(.75) - g.quantile(.25))}).to_dict(
            orient="index")
    return stats


def stage3_num_cat(train):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ["availability_365", "minimum_nights",
                              "calculated_host_listings_count"]):
        sns.boxplot(data=train, x="room_type", y=col, hue="room_type", legend=True,
                    ax=ax, showfliers=False)
        ax.set(title=f"{col} by room_type", xlabel="room_type", ylabel=col)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Figure 7 — Numerical features by room_type (train, outliers not drawn)")
    save(fig, "fig07-numeric-by-room-type.png")
    cols = ["availability_365", "minimum_nights", "calculated_host_listings_count"]
    g = train.groupby("room_type")[cols]
    return {stat: getattr(g, stat)().round(1).to_dict(orient="index")
            for stat in ["median", "mean", "std"]}


def main():
    FIG_DIR.mkdir(exist_ok=True)
    raw = load_raw()
    X_train, X_test, y_train, y_test = split(clean(raw))
    train = X_train.assign(**{TARGET: y_train})
    numbers = {
        "quality": stage1_quality(raw),
        "rows_after_clean": len(X_train) + len(X_test),
        "split": {"train": len(X_train), "test": len(X_test)},
        "test_unseen_neighbourhoods": sorted(
            set(X_test["neighbourhood"]) - set(X_train["neighbourhood"])),
        "target": stage1_target(y_train),
        "numeric": stage2_numeric(X_train),
        "categorical": stage2_categorical(X_train),
        "correlation": stage3_correlation(train),
        "cat_target": stage3_cat_target(train),
        "num_cat": stage3_num_cat(train),
    }
    OUT_JSON.write_text(json.dumps(numbers, indent=2, ensure_ascii=False, default=str))
    print(f"ok → {OUT_JSON}")


if __name__ == "__main__":
    main()
