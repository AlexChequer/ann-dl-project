"""Exercise 1 — Exploring class separability in 2D.

Gera as 4 classes gaussianas do enunciado, salva a figura em ``figures/`` e
imprime as métricas que alimentam a tabela *Results summary* do relatório.

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise1_point_clouds.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1] / "figures"
RNG = np.random.default_rng(42)  # (1)!

CLASSES = {
    0: {"mean": [2.0, 3.0], "std": [0.8, 2.5]},
    1: {"mean": [5.0, 6.0], "std": [1.2, 1.9]},
    2: {"mean": [8.0, 1.0], "std": [0.9, 0.9]},
    3: {"mean": [15.0, 4.0], "std": [0.5, 2.0]},
}
N_PER_CLASS = 100


def generate(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Amostra 100 pontos por classe, com os desvios multiplicados por ``scale``."""
    xs, ys = [], []
    for label, params in CLASSES.items():
        mean = np.asarray(params["mean"])
        std = np.asarray(params["std"]) * scale
        xs.append(RNG.normal(mean, std, size=(N_PER_CLASS, 2)))
        ys.append(np.full(N_PER_CLASS, label))
    return np.vstack(xs), np.concatenate(ys)


def separation_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Distância média entre centróides dividida pela dispersão média intraclasse."""
    centroids = np.stack([X[y == c].mean(axis=0) for c in CLASSES])
    spreads = np.array([np.linalg.norm(X[y == c] - centroids[c], axis=1).mean() for c in CLASSES])
    pairwise = [
        np.linalg.norm(centroids[i] - centroids[j])
        for i in CLASSES
        for j in CLASSES
        if i < j
    ]
    return float(np.mean(pairwise) / spreads.mean())


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    X, y = generate()
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in CLASSES:
        ax.scatter(*X[y == c].T, s=14, alpha=0.75, label=f"Classe {c}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Nuvens de pontos gaussianas (scale = 1.0)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-point-clouds.png", dpi=150)
    plt.close(fig)  # (2)!

    for scale in (0.5, 1.0, 2.0):
        Xs, ys = generate(scale)
        print(f"scale={scale:>4} | separation ratio = {separation_ratio(Xs, ys):.3f}")


if __name__ == "__main__":
    main()
