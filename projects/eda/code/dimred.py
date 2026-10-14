"""Stage 4: fitted pipeline, outlier count and Figures 8 to 10.

Run from the repository root:  python docs/projects/eda/code/dimred.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import umap  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from pipeline import (  # noqa: E402
    CLIP_QUANTILE, NUM_COLS, SEED, SKEWED_COLS, build_preprocessor, load_split,
    transform_target,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_JSON = Path(__file__).resolve().parent / "dimred_numbers.json"
SAMPLE_SIZE = 4000
PERPLEXITIES = [5, 50]
N_NEIGHBORS = [5, 50]
N_TOP_LOADINGS = 5


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=110)
    plt.close(fig)


def scatter(ax, emb, color, title, xlabel, ylabel):
    pts = ax.scatter(emb[:, 0], emb[:, 1], c=color, cmap="viridis", s=4, alpha=0.6,
                     label="listing (color = log1p(price))")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="upper right", fontsize=7, markerscale=3)
    return pts


def outlier_rows(X_train):
    # Same order as the pipeline: impute 0 before computing the p99
    skewed = X_train[SKEWED_COLS].fillna(0)
    return int((skewed > skewed.quantile(CLIP_QUANTILE)).any(axis=1).sum())


def run_pca(Xt, names, color, idx):
    pca = PCA(random_state=SEED).fit(Xt)
    cum = np.cumsum(pca.explained_variance_ratio_)
    emb = pca.transform(Xt[idx])[:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(np.arange(1, len(cum) + 1), cum, marker=".", label="cumulative")
    axes[0].axhline(0.9, ls="--", c="grey", label="90%")
    axes[0].set(title="Cumulative explained variance", xlabel="number of components",
                ylabel="fraction of variance")
    axes[0].legend()
    pts = scatter(axes[1], emb, color, "PCA — PC1 × PC2",
                  f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)",
                  f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
    fig.colorbar(pts, ax=axes[1], label="log1p(price)")
    fig.suptitle("Figure 8 — PCA of the scaled features (train)")
    save(fig, "fig08-pca.png")
    loadings = pd.DataFrame(pca.components_[:2].T, index=names, columns=["PC1", "PC2"])
    top = {pc: loadings[pc].abs().sort_values(ascending=False).head(N_TOP_LOADINGS)
           .index.map(lambda n: f"{n} ({loadings.loc[n, pc]:+.2f})").tolist()
           for pc in ["PC1", "PC2"]}
    return {
        "pc1_pct": round(100 * pca.explained_variance_ratio_[0], 2),
        "pc2_pct": round(100 * pca.explained_variance_ratio_[1], 2),
        "pc1_pc2_pct": round(100 * cum[1], 2),
        "n_components_90": int(np.searchsorted(cum, 0.9) + 1),
        "top_loadings": top,
    }


def run_nonlinear(Xs, color, kind, params, fig_no, fname):
    fig, axes = plt.subplots(1, len(params), figsize=(12, 4.8))
    for ax, p in zip(axes, params):
        if kind == "t-SNE":
            model = TSNE(n_components=2, perplexity=p, random_state=SEED, init="pca")
            label = f"perplexity={p}"
        else:
            model = umap.UMAP(n_neighbors=p, min_dist=0.1, random_state=SEED)
            label = f"n_neighbors={p}"
        emb = model.fit_transform(Xs)
        pts = scatter(ax, emb, color, f"{kind} ({label})", f"{kind} 1", f"{kind} 2")
    fig.colorbar(pts, ax=axes, label="log1p(price)")
    fig.suptitle(f"Figure {fig_no} — {kind} on a sample of {len(Xs)} train listings")
    fig.savefig(FIG_DIR / fname, dpi=110)
    plt.close(fig)


def main():
    X_train, X_test, y_train, y_test = load_split()
    pre = build_preprocessor().fit(X_train)
    Xt_train, Xt_test = pre.transform(X_train), pre.transform(X_test)
    names = list(pre.get_feature_names_out())

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xt_train), size=SAMPLE_SIZE, replace=False)
    color = transform_target(y_train.to_numpy()[idx])

    numbers = {
        "outlier_rows_train": outlier_rows(X_train),
        "outlier_rows_pct": round(100 * outlier_rows(X_train) / len(X_train), 2),
        "clip_upper": X_train[SKEWED_COLS].fillna(0).quantile(CLIP_QUANTILE).round(2).to_dict(),
        "shape_train": list(Xt_train.shape),
        "shape_test": list(Xt_test.shape),
        "nan_train": int(np.isnan(Xt_train).sum()),
        "nan_test": int(np.isnan(Xt_test).sum()),
        "n_numeric": len(NUM_COLS),
        "n_onehot": len(names) - len(NUM_COLS),
        "pca": run_pca(Xt_train, names, color, idx),
    }
    run_nonlinear(Xt_train[idx], color, "t-SNE", PERPLEXITIES, 9, "fig09-tsne.png")
    run_nonlinear(Xt_train[idx], color, "UMAP", N_NEIGHBORS, 10, "fig10-umap.png")
    OUT_JSON.write_text(json.dumps(numbers, indent=2, ensure_ascii=False))
    print(json.dumps(numbers, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
