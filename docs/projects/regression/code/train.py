"""Stage 3: training loop with early stopping, learning curves and the LR comparison.

Run from the repository root:  python docs/projects/regression/code/train.py
Produces Figures 1 and 2 and train_numbers.json. Uses train and validation only.

`train_model` is the single training entry point reused by Stages 4 and 5.
"""

import copy
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from data import (  # noqa: E402
    CODE_DIR, FIG_DIR, SEED, load_data, regression_metrics, to_log, to_price,
)
from model import MLP, count_parameters  # noqa: E402

OUT_JSON = CODE_DIR / "train_numbers.json"

# Base configuration: no regularization. Stage 4 overrides keys of this dict.
BASE_CONFIG = {
    "hidden": (128, 64),
    "dropout": 0.0,
    "batchnorm": False,
    "weight_decay": 0.0,
    "optimizer": "adamw",   # with weight_decay=0, AdamW is plain Adam
    "lr": 1e-3,
    "batch_size": 256,
    "max_epochs": 200,
    "patience": 15,
    "loss": "huber",
}
LR_GRID = [1e-2, 1e-3, 1e-4]

LOSSES = {"huber": nn.HuberLoss, "mse": nn.MSELoss, "mae": nn.L1Loss}


def make_optimizer(model, cfg):
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                               weight_decay=cfg["weight_decay"])
    return torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                             weight_decay=cfg["weight_decay"])


def train_model(X_train, y_train_log, X_val, y_val_log, config=None, seed=SEED):
    """Train an MLP with early stopping on the validation loss.

    Returns (model with the best weights restored, history dict).
    history: train_loss and val_loss per epoch, best_epoch (1-based), stopped_epoch.
    """
    cfg = {**BASE_CONFIG, **(config or {})}
    torch.manual_seed(seed)
    X_tr, y_tr = torch.as_tensor(X_train), torch.as_tensor(y_train_log)
    X_va, y_va = torch.as_tensor(X_val), torch.as_tensor(y_val_log)

    model = MLP(X_tr.shape[1], cfg["hidden"], cfg["dropout"], cfg["batchnorm"])
    optimizer = make_optimizer(model, cfg)
    loss_fn = LOSSES[cfg["loss"]]()

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(X_tr))
        total = 0.0
        for start in range(0, len(order), cfg["batch_size"]):
            idx = order[start:start + cfg["batch_size"]]
            optimizer.zero_grad()
            loss = loss_fn(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va), y_va).item()
        history["train_loss"].append(total / len(X_tr))
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                break

    model.load_state_dict(best_state)  # restore the best weights
    history["best_epoch"] = best_epoch
    history["stopped_epoch"] = epoch
    return model, history


def predict_price(model, X):
    """Predictions in US$ (inverse of log1p)."""
    model.eval()
    with torch.no_grad():
        return to_price(model(torch.as_tensor(X)).numpy())


def plot_learning_curves(history, name):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="validation")
    ax.axvline(history["best_epoch"], color="gray", ls="--",
               label=f"best epoch ({history['best_epoch']})")
    ax.set(title="Huber loss on log1p(price) per epoch — base network",
           xlabel="epoch", ylabel="Huber loss (log scale)", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=110)
    plt.close(fig)


def plot_lr_comparison(histories, name):
    fig, ax = plt.subplots(figsize=(7, 4))
    for lr, h in histories.items():
        ax.plot(np.arange(1, len(h["val_loss"]) + 1), h["val_loss"], label=f"lr = {lr:g}")
    ax.set(title="Validation loss per epoch for three learning rates",
           xlabel="epoch", ylabel="validation Huber loss", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=110)
    plt.close(fig)


def main():
    d = load_data()
    y_tr, y_va = to_log(d["y_train"]), to_log(d["y_val"])

    numbers = {"config": BASE_CONFIG, "lr_comparison": {}}
    histories = {}
    for lr in LR_GRID:
        model, h = train_model(d["X_train"], y_tr, d["X_val"], y_va, {"lr": lr})
        histories[lr] = h
        numbers["lr_comparison"][f"{lr:g}"] = {
            "best_epoch": h["best_epoch"],
            "stopped_epoch": h["stopped_epoch"],
            "best_val_loss": min(h["val_loss"]),
            "val_loss": h["val_loss"],
            "val": regression_metrics(d["y_val"], predict_price(model, d["X_val"])),
        }
        if lr == BASE_CONFIG["lr"]:
            base_model, base_h = model, h

    best = base_h["best_epoch"] - 1
    numbers["base"] = {
        "parameters": count_parameters(base_model),
        "train_rows": len(y_tr),
        "best_epoch": base_h["best_epoch"],
        "stopped_epoch": base_h["stopped_epoch"],
        "train_loss_at_best": base_h["train_loss"][best],
        "val_loss_at_best": base_h["val_loss"][best],
        "train_loss_last": base_h["train_loss"][-1],
        "val_loss_last": base_h["val_loss"][-1],
        "train_loss": base_h["train_loss"],
        "train": regression_metrics(d["y_train"], predict_price(base_model, d["X_train"])),
        "val": regression_metrics(d["y_val"], predict_price(base_model, d["X_val"])),
    }
    plot_learning_curves(base_h, "fig01-learning-curves.png")
    plot_lr_comparison(histories, "fig02-lr-comparison.png")
    OUT_JSON.write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
