
import copy
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model,
    dataloader,
    criterion,
    device,
    optimizer=None,
    grad_clip=1.0
):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0

    for input_ids, target_ids in dataloader:

        input_ids = input_ids.to(
            device, non_blocking=True
        )

        target_ids = target_ids.to(
            device, non_blocking=True
        )

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):

            logits, _ = model(input_ids)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )

            if is_training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip
                )

                optimizer.step()

        valid_tokens = (
            target_ids != criterion.ignore_index
        ).sum().item()

        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError("No valid target tokens.")

    return total_loss / total_tokens


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=5,
    learning_rate=0.001,
    pad_id=0,
    patience=3,
    checkpoint_path="checkpoints/rnn_best.pt",
    figure_path="results/figures/rnn_loss.png",
    model_name="RNN"
):
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    model = model.to(device)

    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_val_loss = float("inf")
    best_epoch = 0
    no_improvement = 0

    Path(checkpoint_path).parent.mkdir(
        parents=True, exist_ok=True
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):

        epoch_start = time.perf_counter()

        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer
        )

        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = (
            time.perf_counter() - epoch_start
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.2f}s",
            flush=True
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            no_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss
                },
                checkpoint_path
            )

            print("  Best model saved.")

        else:
            no_improvement += 1

        if no_improvement >= patience:
            print("Early stopping triggered.")
            break

    if device.type == "cuda":
        torch.cuda.synchronize()

    training_time = time.perf_counter() - start_time

    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )
        print(f"Peak GPU memory: {peak_memory_mb:.2f} MB")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    Path(figure_path).parent.mkdir(
        parents=True, exist_ok=True
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(history["train_loss"]) + 1),
        history["train_loss"],
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        range(1, len(history["val_loss"]) + 1),
        history["val_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{model_name} Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {training_time:.2f}s")

    return model, history, training_time, best_epoch
