
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src.rnn import RNNLanguageModel
from src.lstm import LSTMLanguageModel
from src.transformer import TransformerLanguageModel


# ============================================================
# 1. Configuration
# ============================================================

DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
RESULT_DIR = Path("results/metrics")

RESULT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 32

# Exclude the first two target positions of each window.
TARGET_START = 2

BATCH_SIZE = 64


# ============================================================
# 2. Load configuration and data
# ============================================================

def load_resources():

    with open(
        "configs/config.yaml",
        "r",
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    with open(
        DATA_DIR / "vocab.json",
        "r",
        encoding="utf-8"
    ) as f:
        vocab = json.load(f)

    test_ids = torch.load(
        DATA_DIR / "test.pt",
        map_location="cpu",
        weights_only=True
    )

    assert test_ids.ndim == 1

    return config, vocab, test_ids


# ============================================================
# 3. Create identical evaluation windows
# ============================================================

def create_windows(token_ids):

    # Match the non-overlapping chunking used during training.
    num_windows = (
        (len(token_ids) - 1) // WINDOW_SIZE
    )

    usable_length = num_windows * WINDOW_SIZE

    inputs = token_ids[:usable_length].reshape(
        num_windows,
        WINDOW_SIZE
    )

    targets = token_ids[
        1:usable_length + 1
    ].reshape(
        num_windows,
        WINDOW_SIZE
    )

    # Verify next-token alignment.
    assert torch.equal(
        inputs[:, 1:],
        targets[:, :-1]
    )

    return inputs, targets


# ============================================================
# 4. Load neural models
# ============================================================

def load_neural_model(
    name,
    config,
    vocab_size,
    pad_id,
    device
):

    common = config["model"]

    if name == "RNN":

        model = RNNLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=common["embedding_dim"],
            hidden_dim=common["hidden_dim"],
            num_layers=common["num_layers"],
            dropout=common["dropout"],
            pad_id=pad_id
        )

        checkpoint_path = (
            CHECKPOINT_DIR / "rnn_best.pt"
        )

    elif name == "LSTM":

        model = LSTMLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=common["embedding_dim"],
            hidden_dim=common["hidden_dim"],
            num_layers=common["num_layers"],
            dropout=common["dropout"],
            pad_id=pad_id
        )

        checkpoint_path = (
            CHECKPOINT_DIR / "lstm_best.pt"
        )

    elif name == "Transformer":

        tc = config["transformer"]

        model = TransformerLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=tc["embedding_dim"],
            num_heads=tc["num_heads"],
            num_layers=tc["num_layers"],
            feedforward_dim=tc["feedforward_dim"],
            dropout=tc["dropout"],
            max_seq_length=tc["max_seq_length"],
            pad_id=pad_id
        )

        checkpoint_path = (
            CHECKPOINT_DIR / "transformer_best.pt"
        )

    else:
        raise ValueError(f"Unknown model: {name}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint_path


# ============================================================
# 5. Neural model evaluation
# ============================================================

@torch.inference_mode()
def evaluate_neural(
    model,
    inputs,
    targets,
    device,
    pad_id
):

    total_nll = 0.0
    total_tokens = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for start in range(0, len(inputs), BATCH_SIZE):

        end = start + BATCH_SIZE

        x = inputs[start:end].to(device)
        y = targets[start:end].to(device)

        logits, _ = model(x)

        # Use the same target positions for all models.
        logits = logits[:, TARGET_START:, :]
        y = y[:, TARGET_START:]

        # Sum loss over non-padding tokens.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=pad_id,
            reduction="sum"
        )

        total_nll += loss.item()

        total_tokens += (
            y != pad_id
        ).sum().item()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    return make_metrics(
        total_nll,
        total_tokens,
        elapsed
    )


# ============================================================
# 6. Trigram evaluation on the SAME positions
# ============================================================

def evaluate_ngram(
    model,
    inputs,
    targets,
    pad_id
):

    total_nll = 0.0
    total_tokens = 0

    start_time = time.perf_counter()

    for x, y in zip(inputs, targets):

        x = x.tolist()
        y = y.tolist()

        for t in range(TARGET_START, WINDOW_SIZE):

            target = y[t]

            if target == pad_id:
                continue

            # The two tokens immediately preceding the target.
            # target y[t] corresponds to input x[t+1].
            context = (x[t - 1], x[t])

            prob = model.probability(
                context,
                target
            )

            if prob <= 0:
                raise ValueError(
                    "Invalid trigram probability."
                )

            total_nll -= math.log(prob)
            total_tokens += 1

    elapsed = time.perf_counter() - start_time

    return make_metrics(
        total_nll,
        total_tokens,
        elapsed
    )


# ============================================================
# 7. Common metric calculation
# ============================================================

def make_metrics(total_nll, total_tokens, elapsed):

    if total_tokens == 0:
        raise ValueError("No evaluation tokens.")

    cross_entropy = total_nll / total_tokens

    perplexity = math.exp(cross_entropy)

    throughput = total_tokens / elapsed

    return {
        "cross_entropy": cross_entropy,
        "perplexity": perplexity,
        "evaluated_tokens": total_tokens,
        "evaluation_time_seconds": elapsed,
        "evaluation_tokens_per_second": throughput
    }


# ============================================================
# 8. Main experiment
# ============================================================

def main():

    torch.manual_seed(42)
    np.random.seed(42)

    config, vocab, test_ids = load_resources()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 65)
    print("PART I - UNIFIED LANGUAGE MODEL EVALUATION")
    print("=" * 65)

    print("Device:", device)
    print("Vocabulary size:", len(vocab))
    print("Total test tokens:", len(test_ids))

    assert (
        config["data"]["sequence_length"] == WINDOW_SIZE
    )

    assert (
        config["transformer"]["max_seq_length"]
        == WINDOW_SIZE
    )

    inputs, targets = create_windows(test_ids)

    print("Evaluation windows:", len(inputs))
    print("Window length:", WINDOW_SIZE)

    expected_tokens = (
        len(inputs) * (WINDOW_SIZE - TARGET_START)
    )

    print("Expected evaluated tokens:", expected_tokens)

    results = []

    # --------------------------------------------------------
    # Trigram
    # --------------------------------------------------------

    print("\nEvaluating Trigram...")

    ngram_path = CHECKPOINT_DIR / "ngram.pkl"

    # Only load pickle files created by your own code.
    with open(ngram_path, "rb") as f:
        ngram = pickle.load(f)

    assert ngram.vocab_size == len(vocab)

    ngram_result = evaluate_ngram(
        ngram,
        inputs,
        targets,
        vocab["<pad>"]
    )

    results.append({
        "model": "Trigram",
        **ngram_result,
        "model_size_mb":
            ngram_path.stat().st_size / (1024 ** 2),
        "trainable_parameters": 0
    })

    print(
        f"Trigram PPL: "
        f"{ngram_result['perplexity']:.4f}"
    )

    # --------------------------------------------------------
    # Neural models
    # --------------------------------------------------------

    for name in ["RNN", "LSTM", "Transformer"]:

        print(f"\nEvaluating {name}...")

        model, checkpoint_path = load_neural_model(
            name,
            config,
            len(vocab),
            vocab["<pad>"],
            device
        )

        parameters = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        metrics = evaluate_neural(
            model,
            inputs,
            targets,
            device,
            vocab["<pad>"]
        )

        results.append({
            "model": name,
            **metrics,
            "model_size_mb":
                checkpoint_path.stat().st_size
                / (1024 ** 2),
            "trainable_parameters": parameters
        })

        print(
            f"{name} PPL: "
            f"{metrics['perplexity']:.4f}"
        )

        del model

    # --------------------------------------------------------
    # Verify equal evaluation token counts
    # --------------------------------------------------------

    counts = [
        result["evaluated_tokens"]
        for result in results
    ]

    assert len(set(counts)) == 1, (
        "Models evaluated different token counts."
    )

    assert counts[0] == expected_tokens

    print("\nToken-count consistency: PASSED")

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    df = pd.DataFrame(results)

    df.to_csv(
        RESULT_DIR / "part1_final_comparison.csv",
        index=False
    )

    with open(
        RESULT_DIR / "part1_evaluation_protocol.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "protocol": "common_32_token_windows",
                "window_size": WINDOW_SIZE,
                "target_start": TARGET_START,
                "batch_size": BATCH_SIZE,
                "num_windows": len(inputs),
                "evaluated_tokens": counts[0],
                "context_boundary": "window_only",
                "internal_eos_reset": False,
                "checkpoint_selection": "validation",
                "test_used_for_tuning": False,
                "notes": (
                    "Identical target positions and "
                    "maximum context windows. "
                    "Models retain their original "
                    "training boundary conventions."
                )
            },
            f,
            indent=2
        )

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    print(
        df[
            [
                "model",
                "cross_entropy",
                "perplexity",
                "evaluated_tokens",
                "evaluation_tokens_per_second"
            ]
        ].to_string(index=False)
    )

    print("\nSaved to:")
    print(RESULT_DIR / "part1_final_comparison.csv")

    print("\nUnified evaluation completed!")


if __name__ == "__main__":
    main()
