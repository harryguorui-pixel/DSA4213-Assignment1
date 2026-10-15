
"""
DSA4213 Assignment 1 - Part II
Plot training and validation loss comparisons.

Usage:
    python plot_part2.py

Inputs:
    results/metrics/rnn_history.csv
    results/metrics/part2/rnn_skipgram_training.json
    results/metrics/part2/rnn_glove_training.json

Outputs:
    results/figures/part2/embedding_training_loss_comparison.png
    results/figures/part2/embedding_validation_loss_comparison.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PART1_PATH = Path("results/metrics/rnn_history.csv")

PART2_DIR = Path("results/metrics/part2")
OUTPUT_DIR = Path("results/figures/part2")


def load_histories():
    # Setting A: Part I RNN history
    a = pd.read_csv(PART1_PATH)

    required_columns = {"epoch", "train_loss", "val_loss"}

    assert required_columns.issubset(a.columns)

    # Setting B/C: Part II training JSON
    with open(
        PART2_DIR / "rnn_skipgram_training.json",
        "r",
        encoding="utf-8"
    ) as f:
        b = json.load(f)

    with open(
        PART2_DIR / "rnn_glove_training.json",
        "r",
        encoding="utf-8"
    ) as f:
        c = json.load(f)

    histories = {
        "A: Random Trainable": {
            "epoch": a["epoch"].tolist(),
            "train_loss": a["train_loss"].tolist(),
            "val_loss": a["val_loss"].tolist()
        },
        "B: Skip-gram Frozen": {
            "epoch": list(range(1, len(b["train_loss"]) + 1)),
            "train_loss": b["train_loss"],
            "val_loss": b["val_loss"]
        },
        "C: GloVe Frozen": {
            "epoch": list(range(1, len(c["train_loss"]) + 1)),
            "train_loss": c["train_loss"],
            "val_loss": c["val_loss"]
        }
    }

    for name, history in histories.items():
        assert len(history["epoch"]) == 5
        assert len(history["train_loss"]) == 5
        assert len(history["val_loss"]) == 5

        print(
            f"{name}: "
            f"Final Train={history['train_loss'][-1]:.4f}, "
            f"Final Val={history['val_loss'][-1]:.4f}"
        )

    return histories


def plot_comparison(histories, metric, title, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.5))

    for name, history in histories.items():
        plt.plot(
            history["epoch"],
            history[metric],
            marker="o",
            linewidth=2,
            markersize=6,
            label=name
        )

    plt.title(title, fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename

    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    print("=" * 65)
    print("PART II - EMBEDDING LOSS COMPARISON")
    print("=" * 65)

    histories = load_histories()

    plot_comparison(
        histories,
        metric="train_loss",
        title="Training Loss: Embedding Variants",
        filename="embedding_training_loss_comparison.png"
    )

    plot_comparison(
        histories,
        metric="val_loss",
        title="Validation Loss: Embedding Variants",
        filename="embedding_validation_loss_comparison.png"
    )

    print("\nLOSS COMPARISON COMPLETED")


if __name__ == "__main__":
    main()
