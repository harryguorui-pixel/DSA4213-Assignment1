
"""
Part II setup validation.

Run from the project root:
    python -m tests.test_part2_setup

This script does not modify Part I files.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.rnn import RNNLanguageModel
from src.train import set_seed


CONFIG_PATH = Path("configs/part2.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_existing_data(config):
    print("\n[1] Checking existing data...")

    data_dir = Path(config["data"]["processed_dir"])

    vocab_path = data_dir / "vocab.json"

    assert vocab_path.exists(), "Missing vocab.json"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    assert len(vocab) == 10000
    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert vocab["<eos>"] == 2

    datasets = {}

    for split in ["train", "validation", "test"]:
        path = data_dir / f"{split}.pt"

        assert path.exists(), f"Missing {path}"

        tokens = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        assert tokens.ndim == 1
        assert tokens.dtype == torch.long
        assert len(tokens) > 33

        assert tokens.min().item() >= 0
        assert tokens.max().item() < len(vocab)

        datasets[split] = tokens

        print(f"  {split}: {len(tokens):,} tokens")

    print(f"  Vocabulary size: {len(vocab):,}")
    print("  Data checks: PASSED")

    return vocab, datasets


def build_model(config, vocab_size):
    params = config["model"]

    return RNNLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        pad_id=0
    )


def check_checkpoint(config, vocab):
    print("\n[2] Checking Part I RNN checkpoint...")

    checkpoint_path = Path(
        config["paths"]["part1_rnn_checkpoint"]
    )

    assert checkpoint_path.exists(), (
        f"Missing checkpoint: {checkpoint_path}"
    )

    model = build_model(config, len(vocab))

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    assert total_params == 3793232, (
        f"Unexpected parameter count: {total_params}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    assert "model_state_dict" in checkpoint

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    print(f"  Total parameters: {total_params:,}")
    print("  Checkpoint loading: PASSED")

    return model


def check_embedding_freezing(config, vocab):
    print("\n[3] Checking frozen embeddings...")

    set_seed(config["project"]["seed"])

    model = build_model(config, len(vocab))

    embedding_dim = config["model"]["embedding_dim"]

    generator = torch.Generator().manual_seed(42)

    fake_embeddings = torch.randn(
        len(vocab),
        embedding_dim,
        generator=generator
    ) * 0.1

    fake_embeddings[vocab["<pad>"]].zero_()

    with torch.no_grad():
        model.embedding.weight.copy_(fake_embeddings)

    model.embedding.weight.requires_grad_(False)

    assert not model.embedding.weight.requires_grad

    assert model.rnn.weight_ih_l0.requires_grad
    assert model.output_layer.weight.requires_grad

    original_embedding = (
        model.embedding.weight.detach().clone()
    )

    original_rnn_weight = (
        model.rnn.weight_ih_l0.detach().clone()
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    input_ids = torch.randint(
        3, len(vocab), (2, 4)
    )

    target_ids = torch.randint(
        3, len(vocab), (2, 4)
    )

    optimizer.zero_grad(set_to_none=True)

    logits, _ = model(input_ids)

    loss = criterion(
        logits.reshape(-1, len(vocab)),
        target_ids.reshape(-1)
    )

    loss.backward()

    assert model.embedding.weight.grad is None
    assert model.rnn.weight_ih_l0.grad is not None

    optimizer.step()

    assert torch.equal(
        original_embedding,
        model.embedding.weight.detach()
    ), "Frozen embedding changed!"

    assert not torch.equal(
        original_rnn_weight,
        model.rnn.weight_ih_l0.detach()
    ), "RNN weights did not update!"

    print("  Embedding requires_grad=False: PASSED")
    print("  Embedding unchanged after update: PASSED")
    print("  RNN weights updated: PASSED")


def check_identical_initialization(config, vocab):
    print("\n[4] Checking identical B/C initialization...")

    seed = config["project"]["seed"]

    set_seed(seed)
    model_b = build_model(config, len(vocab))

    set_seed(seed)
    model_c = build_model(config, len(vocab))

    for (name_b, param_b), (name_c, param_c) in zip(
        model_b.named_parameters(),
        model_c.named_parameters()
    ):
        assert name_b == name_c
        assert torch.equal(param_b, param_c), (
            f"Initialization differs: {name_b}"
        )

    print("  Identical initial weights: PASSED")


def main():
    print("=" * 60)
    print("DSA4213 PART II - SETUP VALIDATION")
    print("=" * 60)

    config = load_config()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if config["training"]["device"] == "cuda":
        assert torch.cuda.is_available(), (
            "CUDA requested but not available"
        )
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    vocab, datasets = load_existing_data(config)

    check_checkpoint(config, vocab)

    check_embedding_freezing(config, vocab)

    check_identical_initialization(config, vocab)

    print("\n" + "=" * 60)
    print("ALL PART II SETUP CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
