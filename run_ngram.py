
import json
import pickle
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.data_utils import tokenize, encode_tokens, decode_tokens
from src.ngram import TrigramLanguageModel


def load_processed_data():

    processed_dir = Path("data/processed")

    with open(
        processed_dir / "vocab.json",
        "r",
        encoding="utf-8"
    ) as f:
        token_to_id = json.load(f)

    id_to_token = {
        idx: token
        for token, idx in token_to_id.items()
    }

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

    test_ids = torch.load(
        processed_dir / "test.pt",
        map_location="cpu",
        weights_only=True
    ).tolist()

    return (
        train_ids,
        val_ids,
        test_ids,
        token_to_id,
        id_to_token
    )


def main():

    with open(
        "configs/config.yaml",
        "r",
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    seed = config["project"]["seed"]

    (
        train_ids,
        val_ids,
        test_ids,
        token_to_id,
        id_to_token
    ) = load_processed_data()

    vocab_size = len(token_to_id)

    model = TrigramLanguageModel(
        vocab_size=vocab_size,
        k=0.1,
        pad_id=token_to_id["<pad>"],
        eos_id=token_to_id["<eos>"]
    )

    print("=" * 60)
    print("Training Trigram Language Model")
    print("=" * 60)

    start_time = time.perf_counter()

    model.fit(train_ids)

    training_time = (
        time.perf_counter() - start_time
    )

    print(f"\nTraining time: {training_time:.2f} seconds")

    print("\nEvaluating validation set...")

    val_results = model.evaluate(val_ids)

    print("Validation cross-entropy:", val_results["cross_entropy"])
    print("Validation perplexity:", val_results["perplexity"])

    print("\nEvaluating test set...")

    test_start = time.perf_counter()

    test_results = model.evaluate(test_ids)

    inference_time = (
        time.perf_counter() - test_start
    )

    throughput = (
        test_results["evaluated_tokens"] / inference_time
    )

    print("Test cross-entropy:", test_results["cross_entropy"])
    print("Test perplexity:", test_results["perplexity"])
    print("Evaluation throughput:", throughput, "tokens/s")

    Path("checkpoints").mkdir(exist_ok=True)

    model_path = Path("checkpoints/ngram.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    model_size_mb = (
        model_path.stat().st_size / (1024 ** 2)
    )

    print(f"Model size: {model_size_mb:.2f} MB")

    Path("results/metrics").mkdir(
        parents=True,
        exist_ok=True
    )

    metrics = {
        "model": "Trigram Add-k",
        "smoothing_k": 0.1,
        "validation_cross_entropy": val_results["cross_entropy"],
        "validation_perplexity": val_results["perplexity"],
        "test_cross_entropy": test_results["cross_entropy"],
        "test_perplexity": test_results["perplexity"],
        "training_time_seconds": training_time,
        "model_size_mb": model_size_mb,
        "evaluation_time_seconds": inference_time,
        "evaluation_tokens_per_second": throughput,
        "evaluated_test_tokens": test_results["evaluated_tokens"],
        **model.model_statistics(),
    }

    df = pd.DataFrame([metrics])

    df.to_csv(
        "results/metrics/ngram_results.csv",
        index=False
    )

    print("\nGenerating sample text...")

    prompts = [
        "the future of technology",
        "in the middle of the night",
        "scientists have discovered"
    ]

    output_lines = []

    for prompt in prompts:

        prompt_tokens = tokenize(prompt)

        prompt_ids = encode_tokens(
            prompt_tokens,
            token_to_id
        )

        generated_ids = model.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=50,
            strategy="greedy",
            seed=seed
        )

        generated_tokens = decode_tokens(
            generated_ids,
            id_to_token
        )

        generated_text = " ".join(generated_tokens)

        print("\nPrompt:", prompt)
        print("Generated:", generated_text)

        output_lines.append(
            f"Prompt: {prompt}\n"
            f"Generated: {generated_text}\n"
        )

    Path("results/generated_text").mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        "results/generated_text/ngram_generations.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(output_lines))

    print("\nAll n-gram experiments completed.")


if __name__ == "__main__":
    main()
