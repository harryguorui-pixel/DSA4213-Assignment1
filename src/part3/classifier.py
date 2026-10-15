import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.data_utils import set_seed


def load_config():
    with open(
        "configs/part3.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def load_features(feature_dir, setting, split):
    path = feature_dir / f"{setting}_{split}.pt"

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    features = data["features"].float()
    labels = data["labels"].long()

    assert features.ndim == 2
    assert labels.ndim == 1
    assert features.size(0) == labels.size(0)
    assert torch.isfinite(features).all()

    return features, labels


def create_loader(features, labels, batch_size, shuffle):
    dataset = TensorDataset(features, labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(features)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


def train_classifier(setting, config, device):
    seed = config["project"]["seed"]

    # Reset seed so each classifier starts from
    # the same initialization and batch ordering.
    set_seed(seed)

    feature_dir = Path("data/part3/features")

    train_x, train_y = load_features(
        feature_dir, setting, "train"
    )
    val_x, val_y = load_features(
        feature_dir, setting, "validation"
    )

    batch_size = config["data"]["batch_size"]

    train_loader = create_loader(
        train_x, train_y, batch_size, shuffle=True
    )
    val_loader = create_loader(
        val_x, val_y, batch_size, shuffle=False
    )

    model = nn.Linear(
        config["model"]["hidden_dim"],
        config["model"]["num_classes"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]

    checkpoint_dir = Path(
        config["paths"]["checkpoint_dir"]
    )
    metrics_dir = Path(
        config["paths"]["metrics_dir"]
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        checkpoint_dir / f"classifier_{setting}_best.pt"
    )

    best_val_loss = float("inf")
    best_epoch = 0
    no_improvement = 0

    history = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                criterion,
                device,
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "setting": setting,
            }, checkpoint_path)

            print("  Best classifier saved.")
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print("Early stopping triggered.")
            break

    if device.type == "cuda":
        torch.cuda.synchronize()

    training_time = time.perf_counter() - start_time

    history_path = (
        metrics_dir / f"{setting}_classifier_history.csv"
    )

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Training time: {training_time:.2f}s")

    return {
        "setting": setting,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_time": training_time,
    }


def main():
    config = load_config()

    requested_device = config["training"]["device"]

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    device = torch.device(requested_device)

    print("=" * 65)
    print("PART III - LINEAR CLASSIFIER TRAINING")
    print("=" * 65)
    print("Device:", device)

    results = []

    for setting in ["random", "skipgram", "glove"]:
        print("\n" + "-" * 65)
        print("Setting:", setting.upper())

        result = train_classifier(
            setting,
            config,
            device,
        )
        results.append(result)

    metrics_dir = Path(config["paths"]["metrics_dir"])

    with open(
        metrics_dir / "classifier_training_summary.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "best_epoch",
                "best_val_loss",
                "training_time",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nAll classifiers trained successfully.")


if __name__ == "__main__":
    main()