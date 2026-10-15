
"""
DSA4213 Assignment 1 - Part II
Unified evaluation for three RNN embedding variants.

Usage:
    python evaluate_part2.py
    python evaluate_part2.py --prompts-file configs/part2_prompts.json

The evaluation protocol is imported directly from Part I.
No models are retrained.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from evaluate_all import (
    WINDOW_SIZE,
    TARGET_START,
    BATCH_SIZE,
    create_windows,
    evaluate_neural
)

from src.rnn import RNNLanguageModel
from src.generate import generate_text


CONFIG_PATH = Path("configs/part2.yaml")

SETTINGS = {
    "A": {
        "name": "Random Trainable",
        "checkpoint": "checkpoints/rnn_best.pt",
        "embedding_file": None,
        "frozen": False
    },
    "B": {
        "name": "Skip-gram Frozen",
        "checkpoint": "checkpoints/part2/rnn_skipgram_best.pt",
        "embedding_file": "data/embeddings/skipgram_100d.npy",
        "frozen": True
    },
    "C": {
        "name": "GloVe Frozen",
        "checkpoint": "checkpoints/part2/rnn_glove_best.pt",
        "embedding_file": "data/embeddings/glove_100d.npy",
        "frozen": True
    }
}


def load_resources():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data"]["processed_dir"])

    with open(data_dir / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    test_ids = torch.load(
        data_dir / "test.pt",
        map_location="cpu",
        weights_only=True
    )

    assert len(vocab) == 10000
    assert config["data"]["sequence_length"] == WINDOW_SIZE
    assert WINDOW_SIZE == 32
    assert TARGET_START == 2
    assert BATCH_SIZE == 64

    return config, vocab, test_ids


def build_model(config, vocab):
    params = config["model"]

    return RNNLanguageModel(
        vocab_size=len(vocab),
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        pad_id=vocab["<pad>"]
    )


def load_model(setting, config, vocab, device):
    info = SETTINGS[setting]
    checkpoint_path = Path(info["checkpoint"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model = build_model(config, vocab)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    if info["frozen"]:
        matrix = np.load(
            info["embedding_file"],
            allow_pickle=False
        )

        saved_weights = (
            model.embedding.weight.detach().cpu().numpy()
        )

        assert np.array_equal(matrix, saved_weights), (
            f"Setting {setting}: embedding mismatch!"
        )

        model.embedding.weight.requires_grad_(False)

        print(f"Setting {setting}: frozen embedding VERIFIED")

    else:
        print("Setting A: original Part I RNN loaded")

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    assert total_params == 3793232

    expected_trainable = (
        3793232 if setting == "A" else 2793232
    )

    assert trainable_params == expected_trainable

    model = model.to(device)
    model.eval()

    return model, checkpoint_path, trainable_params


def verify_part1_baseline(metrics):
    """
    Compare Setting A with the original Part I result.
    Evaluation time is not compared because it varies.
    """
    path = Path("results/metrics/part1_final_comparison.csv")

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    baseline = df.loc[df["model"] == "RNN"]

    assert len(baseline) == 1

    original_ce = float(baseline.iloc[0]["cross_entropy"])
    original_ppl = float(baseline.iloc[0]["perplexity"])

    assert math.isclose(
        metrics["cross_entropy"],
        original_ce,
        rel_tol=0,
        abs_tol=1e-4
    ), "Setting A CE differs from Part I!"

    assert math.isclose(
        metrics["perplexity"],
        original_ppl,
        rel_tol=0,
        abs_tol=0.02
    ), "Setting A PPL differs from Part I!"

    print("Setting A vs Part I baseline: PASSED")


def load_prompts(path):
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    if not isinstance(prompts, list) or len(prompts) != 3:
        raise ValueError("Provide exactly three prompts.")

    if not all(
        isinstance(p, str) and p.strip()
        for p in prompts
    ):
        raise ValueError("Prompts must be non-empty strings.")

    return prompts


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50
    )

    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")

    torch.manual_seed(42)
    np.random.seed(42)

    config, vocab, test_ids = load_resources()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    inputs, targets = create_windows(test_ids)

    expected_tokens = (
        len(inputs) * (WINDOW_SIZE - TARGET_START)
    )

    assert len(inputs) == 7893
    assert expected_tokens == 236790

    print("=" * 65)
    print("PART II - UNIFIED EMBEDDING ABLATION EVALUATION")
    print("=" * 65)

    print(f"Device: {device}")
    print(f"Test tokens: {len(test_ids):,}")
    print(f"Windows: {len(inputs):,}")
    print(f"Expected evaluated tokens: {expected_tokens:,}")

    id_to_token = {
        int(idx): token
        for token, idx in vocab.items()
    }

    prompts = load_prompts(args.prompts_file)

    results = []
    generated = {}

    for setting, info in SETTINGS.items():
        print("\n" + "-" * 65)
        print(f"Setting {setting}: {info['name']}")
        print("-" * 65)

        model, checkpoint_path, trainable_params = (
            load_model(setting, config, vocab, device)
        )

        metrics = evaluate_neural(
            model=model,
            inputs=inputs,
            targets=targets,
            device=device,
            pad_id=vocab["<pad>"]
        )

        assert metrics["evaluated_tokens"] == expected_tokens
        assert math.isfinite(metrics["cross_entropy"])
        assert math.isfinite(metrics["perplexity"])

        if setting == "A":
            verify_part1_baseline(metrics)

        total_params = sum(
            p.numel() for p in model.parameters()
        )

        result = {
            "setting": setting,
            "embedding": info["name"],
            **metrics,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "checkpoint_size_mb": (
                checkpoint_path.stat().st_size / (1024 ** 2)
            )
        }

        results.append(result)

        print(f"Test CE: {metrics['cross_entropy']:.6f}")
        print(f"Test PPL: {metrics['perplexity']:.4f}")
        print(f"Evaluated tokens: {metrics['evaluated_tokens']:,}")

        if prompts is not None:
            generated[setting] = {
                "embedding": info["name"],
                "generations": []
            }

            for prompt in prompts:
                text = generate_text(
                    model=model,
                    prompt=prompt,
                    token_to_id=vocab,
                    id_to_token=id_to_token,
                    device=device,
                    max_new_tokens=args.max_new_tokens
                )

                generated[setting]["generations"].append({
                    "prompt": prompt,
                    "generated_text": text
                })

                print(f"\nPrompt: {prompt}")
                print(f"Generated: {text}")

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    counts = [r["evaluated_tokens"] for r in results]

    assert len(set(counts)) == 1
    assert counts[0] == expected_tokens

    output_dir = Path(config["paths"]["metrics_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = (
        output_dir / "part2_final_comparison.csv"
    )

    if comparison_path.exists():
        raise FileExistsError(comparison_path)

    df = pd.DataFrame(results)

    df.to_csv(comparison_path, index=False)

    protocol = {
        "protocol": "common_32_token_windows",
        "window_size": WINDOW_SIZE,
        "target_start": TARGET_START,
        "batch_size": BATCH_SIZE,
        "evaluation_windows": len(inputs),
        "evaluated_tokens": expected_tokens,
        "context_boundary": "window_only",
        "internal_eos_reset": False,
        "checkpoint_selection": "validation",
        "test_used_for_tuning": False,
        "baseline": "Part I RNN",
        "same_protocol_as_part1": True
    }

    protocol_path = (
        output_dir / "part2_evaluation_protocol.json"
    )

    if protocol_path.exists():
        raise FileExistsError(protocol_path)

    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    if prompts is not None:
        generated_dir = Path(
            config["paths"]["generated_dir"]
        )

        generated_dir.mkdir(parents=True, exist_ok=True)

        generated_path = (
            generated_dir / "part2_generated_text.json"
        )

        if generated_path.exists():
            raise FileExistsError(generated_path)

        with open(
            generated_path, "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "decoding": "greedy",
                    "max_new_tokens": args.max_new_tokens,
                    "results": generated
                },
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"\nGenerated text saved: {generated_path}")

    print("\n" + "=" * 65)
    print("FINAL PART II COMPARISON")
    print("=" * 65)

    print(
        df[
            [
                "setting",
                "embedding",
                "cross_entropy",
                "perplexity",
                "evaluated_tokens"
            ]
        ].to_string(index=False)
    )

    print("\nToken-count consistency: PASSED")
    print(f"Saved comparison: {comparison_path}")
    print("PART II EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
