
"""
DSA4213 Assignment 1 - Part II
Train RNN with frozen pretrained embeddings.

Usage:
    python run_part2.py --setting B
    python run_part2.py --setting C
    python run_part2.py --setting all
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data_utils import create_dataloaders
from src.rnn import RNNLanguageModel
from src.train import set_seed, train_model


CONFIG_PATH = Path("configs/part2.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_data(config):
    data_dir = Path(config["data"]["processed_dir"])

    with open(
        data_dir / "vocab.json",
        "r",
        encoding="utf-8"
    ) as f:
        vocab = json.load(f)

    encoded = {}

    for split in ["train", "validation"]:
        encoded[split] = torch.load(
            data_dir / f"{split}.pt",
            map_location="cpu",
            weights_only=True
        )

    return vocab, encoded


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


def load_embedding(setting, config, vocab):
    embedding_dir = Path(config["paths"]["embedding_dir"])

    filename = {
        "B": "skipgram_100d.npy",
        "C": "glove_100d.npy"
    }[setting]

    path = embedding_dir / filename

    if not path.exists():
        raise FileNotFoundError(path)

    matrix = np.load(path, allow_pickle=False)

    expected_shape = (
        len(vocab),
        config["model"]["embedding_dim"]
    )

    assert matrix.shape == expected_shape
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()

    assert np.all(matrix[vocab["<pad>"]] == 0)

    metadata_path = embedding_dir / (
        "skipgram_metadata.json"
        if setting == "B"
        else "glove_metadata.json"
    )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    vocab_hash = sha256_file(
        Path(config["data"]["processed_dir"]) / "vocab.json"
    )

    assert metadata["vocab_sha256"] == vocab_hash, (
        "Vocabulary hash mismatch!"
    )

    print(f"Loaded embedding: {path}")
    print(f"Embedding shape: {matrix.shape}")
    print("Vocabulary hash: VERIFIED")

    return torch.from_numpy(matrix.copy())


def prepare_model(setting, config, vocab, device):
    # Reset seed before model construction.
    set_seed(config["project"]["seed"])

    model = build_model(config, len(vocab))

    # The RNN and output layer are still freshly initialized.
    initial_rnn = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if not name.startswith("embedding.")
    }

    matrix = load_embedding(setting, config, vocab)

    with torch.no_grad():
        model.embedding.weight.copy_(matrix)

    model.embedding.weight.requires_grad_(False)

    assert not model.embedding.weight.requires_grad

    for name, param in model.named_parameters():
        if not name.startswith("embedding."):
            assert param.requires_grad
            assert torch.equal(param, initial_rnn[name])

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params-trainable_params:,}")

    assert total_params == 3793232
    assert total_params - trainable_params == len(vocab) * 100

    model = model.to(device)

    original_embedding = (
        model.embedding.weight.detach().cpu().clone()
    )

    return model, original_embedding


def train_setting(setting, config):
    name = {
        "B": "Skip-gram Frozen",
        "C": "GloVe Frozen"
    }[setting]

    print("\n" + "=" * 65)
    print(f"PART II - SETTING {setting}: {name}")
    print("=" * 65)

    device = torch.device(config["training"]["device"])

    assert device.type == "cuda"
    assert torch.cuda.is_available()

    print(f"Device: {torch.cuda.get_device_name(0)}")

    set_seed(config["project"]["seed"])

    vocab, encoded = load_data(config)

    model, original_embedding = prepare_model(
        setting,
        config,
        vocab,
        device
    )

    loaders = create_dataloaders(
        encoded,
        sequence_length=config["data"]["sequence_length"],
        batch_size=config["data"]["batch_size"]
    )

    checkpoint_dir = Path(
        config["paths"]["checkpoint_dir"]
    )

    figure_dir = Path(
        config["paths"]["figure_dir"]
    )

    metrics_dir = Path(
        config["paths"]["metrics_dir"]
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    filename = {
        "B": "rnn_skipgram",
        "C": "rnn_glove"
    }[setting]

    checkpoint_path = checkpoint_dir / f"{filename}_best.pt"
    figure_path = figure_dir / f"{filename}_loss.png"
    metrics_path = metrics_dir / f"{filename}_training.json"

    # Prevent accidentally overwriting an existing experiment.
    for path in [checkpoint_path, figure_path, metrics_path]:
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}\n"
                "Move or back up existing results before rerunning."
            )

    print("\nStarting training...")

    model, history, training_time, best_epoch = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["validation"],
        device=device,
        epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        pad_id=vocab["<pad>"],
        patience=config["training"]["patience"],
        checkpoint_path=str(checkpoint_path),
        figure_path=str(figure_path),
        model_name=f"Part II - {name}"
    )

    # train_model reloads the best checkpoint.
    trained_embedding = (
        model.embedding.weight.detach().cpu()
    )

    assert torch.equal(
        original_embedding,
        trained_embedding
    ), "Frozen embedding changed during training!"

    print("\nFrozen embedding verification: PASSED")

    results = {
        "setting": setting,
        "embedding": name,
        "seed": config["project"]["seed"],
        "epochs_completed": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_validation_loss": min(history["val_loss"]),
        "training_time_seconds": training_time,
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "embedding_frozen": True,
        "checkpoint_path": str(checkpoint_path),
        "figure_path": str(figure_path)
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Loss figure: {figure_path}")
    print(f"Metrics: {metrics_path}")

    print(f"\nSetting {setting} completed successfully.")

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--setting",
        choices=["B", "C", "all"],
        required=True
    )

    args = parser.parse_args()

    config = load_config()

    settings = (
        ["B", "C"]
        if args.setting == "all"
        else [args.setting]
    )

    for setting in settings:
        train_setting(setting, config)

    print("\n" + "=" * 65)
    print("REQUESTED PART II TRAINING COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
