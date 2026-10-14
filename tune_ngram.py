
import json
import time
from pathlib import Path

import pandas as pd
import torch

from src.ngram import TrigramLanguageModel


def main():

    processed_dir = Path("data/processed")
    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    with open(
        processed_dir / "vocab.json",
        "r",
        encoding="utf-8"
    ) as f:
        token_to_id = json.load(f)

    # Load training and validation tokens
    train_ids = torch.load(
        processed_dir / "train.pt",
        map_location="cpu",
        weights_only=True
    ).tolist()

    val_ids = torch.load(
        processed_dir / "validation.pt",
        map_location="cpu",
        weights_only=True
    ).tolist()

    # Candidate smoothing parameters
    k_values = [0.1, 0.01, 0.001]

    results = []

    for k in k_values:

        print(f"\n{'=' * 50}")
        print(f"Testing smoothing k = {k}")
        print("=" * 50)

        model = TrigramLanguageModel(
            vocab_size=len(token_to_id),
            k=k,
            pad_id=token_to_id["<pad>"],
            eos_id=token_to_id["<eos>"]
        )

        start = time.perf_counter()
        model.fit(train_ids)
        train_time = time.perf_counter() - start

        val_metrics = model.evaluate(val_ids)

        print(
            f"Validation PPL: "
            f"{val_metrics['perplexity']:.4f}"
        )

        results.append({
            "k": k,
            "validation_cross_entropy":
                val_metrics["cross_entropy"],
            "validation_perplexity":
                val_metrics["perplexity"],
            "training_time_seconds": train_time
        })

    # Save all results
    df = pd.DataFrame(results)

    df.to_csv(
        results_dir / "ngram_tuning.csv",
        index=False
    )

    print("\nSmoothing comparison:")
    print(df.to_string(index=False))

    # Select best k using validation PPL only
    best_row = df.loc[
        df["validation_perplexity"].idxmin()
    ]

    best_k = float(best_row["k"])

    print("\n" + "=" * 50)
    print(f"Best smoothing k: {best_k}")
    print(
        "Best Validation PPL:",
        round(
            best_row["validation_perplexity"],
            4
        )
    )
    print("=" * 50)

    # Save best configuration
    with open(
        results_dir / "ngram_best_config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "best_k": best_k,
                "selection_metric": "validation_perplexity"
            },
            f,
            indent=2
        )


if __name__ == "__main__":
    main()
