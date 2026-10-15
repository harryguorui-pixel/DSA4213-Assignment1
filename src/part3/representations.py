import json
import time
from pathlib import Path

import torch
import yaml
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from src.data_utils import set_seed
from src.rnn import RNNLanguageModel


def load_config():
    with open(
        "configs/part3.yaml",
        "r",
        encoding="utf-8"
    ) as f:
        return yaml.safe_load(f)


def load_frozen_rnn(checkpoint_path, config, device):
    """
    Load a trained RNN and freeze all its parameters.
    """
    model_config = config["model"]

    model = RNNLanguageModel(
        vocab_size=10000,
        embedding_dim=model_config["embedding_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        pad_id=0,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    assert not any(
        p.requires_grad for p in model.parameters()
    ), "Language model is not fully frozen."

    print(
        f"Loaded: {checkpoint_path} | "
        f"Epoch: {checkpoint['epoch']} | "
        f"Val Loss: {checkpoint['val_loss']:.4f}"
    )

    return model


@torch.no_grad()
def extract_features(model, data_path, batch_size, device):
    """
    Extract final valid hidden states from the last RNN layer.

    Output:
        features: [N, hidden_dim]
        labels:   [N]
    """
    data = torch.load(
        data_path,
        map_location="cpu",
        weights_only=True,
    )

    input_ids = data["input_ids"]
    labels = data["labels"]
    lengths = data["lengths"]

    assert input_ids.ndim == 2
    assert labels.ndim == 1
    assert lengths.ndim == 1

    assert input_ids.size(0) == labels.size(0)
    assert input_ids.size(0) == lengths.size(0)

    assert torch.all(lengths >= 1)
    assert torch.all(lengths <= input_ids.size(1))

    dataset = TensorDataset(
        input_ids,
        labels,
        lengths,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    feature_batches = []
    label_batches = []

    for batch_ids, batch_labels, batch_lengths in loader:
        batch_ids = batch_ids.to(device)

        # Embedding: [B, T, 100]
        embedded = model.embedding(batch_ids)

        # Exclude padding from recurrent computation.
        packed = pack_padded_sequence(
            embedded,
            batch_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Hidden: [num_layers, B, hidden_dim]
        _, hidden = model.rnn(packed)

        # Final hidden state of the last RNN layer.
        features = hidden[-1]

        feature_batches.append(
            features.cpu()
        )

        label_batches.append(
            batch_labels.cpu()
        )

    features = torch.cat(feature_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)

    assert features.shape == (
        len(dataset),
        model.hidden_dim,
    )

    assert torch.isfinite(features).all(), (
        "Features contain NaN or Inf."
    )

    return features, labels


def main():
    config = load_config()
    set_seed(config["project"]["seed"])

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    paths = config["paths"]

    output_dir = Path(
        "data/part3/features"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoints = {
        "random": paths["checkpoint_random"],
        "skipgram": paths["checkpoint_skipgram"],
        "glove": paths["checkpoint_glove"],
    }

    processed_dir = Path(
        paths["processed_dir"]
    )

    splits = [
        "train",
        "validation",
        "test",
    ]

    metadata = {
        "device": str(device),
        "representation": "final_valid_hidden",
        "hidden_dim": config["model"]["hidden_dim"],
        "settings": {},
    }

    print("=" * 65)
    print("PART III - REPRESENTATION EXTRACTION")
    print("=" * 65)
    print("Device:", device)

    for setting, checkpoint_path in checkpoints.items():
        print("\n" + "-" * 65)
        print("Setting:", setting.upper())

        model = load_frozen_rnn(
            checkpoint_path,
            config,
            device,
        )

        metadata["settings"][setting] = {}

        for split in splits:
            data_path = (
                processed_dir / f"{split}.pt"
            )

            start_time = time.perf_counter()

            features, labels = extract_features(
                model=model,
                data_path=data_path,
                batch_size=config["data"]["batch_size"],
                device=device,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time

            output_path = (
                output_dir /
                f"{setting}_{split}.pt"
            )

            torch.save(
                {
                    "features": features,
                    "labels": labels,
                },
                output_path,
            )

            metadata["settings"][setting][split] = {
                "samples": int(features.size(0)),
                "feature_dim": int(features.size(1)),
                "extraction_seconds": elapsed,
            }

            print(
                f"{split:12s} | "
                f"Features: {tuple(features.shape)} | "
                f"Time: {elapsed:.2f}s"
            )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\nRepresentation extraction completed.")
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()