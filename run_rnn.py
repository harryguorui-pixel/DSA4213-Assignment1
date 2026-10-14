
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.data_utils import (
    create_dataloaders
)
from src.rnn import RNNLanguageModel
from src.train import train_model, set_seed
from src.evaluate import (
    evaluate_model,
    count_parameters
)
from src.generate import generate_text


def main():

    # 1. Load configuration
    with open(
        "configs/config.yaml",
        "r",
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    set_seed(config["project"]["seed"])

    # 2. Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("DSA4213 - RNN Language Model")
    print("=" * 60)

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # 3. Load vocabulary
    processed_dir = Path(
        config["paths"]["processed_dir"]
    )

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

    # 4. Load tokenized data
    encoded = {}

    for split in ["train", "validation", "test"]:

        encoded[split] = torch.load(
            processed_dir / f"{split}.pt",
            map_location="cpu",
            weights_only=True
        ).tolist()

    loaders = create_dataloaders(
        encoded,
        sequence_length=config["data"]["sequence_length"],
        batch_size=config["data"]["batch_size"]
    )

    # 5. Create model
    model = RNNLanguageModel(
        vocab_size=len(token_to_id),
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        pad_id=token_to_id["<pad>"]
    )

    parameters = count_parameters(model)

    print(f"Trainable parameters: {parameters:,}")

    # 6. Train
    model, history, training_time, best_epoch = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["validation"],
        device=device,
        epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        pad_id=token_to_id["<pad>"],
        patience=3,
        checkpoint_path="checkpoints/rnn_best.pt",
        figure_path="results/figures/rnn_loss.png"
    )

    # 7. Evaluate
    print("\nEvaluating validation set...")

    val_results = evaluate_model(
        model,
        loaders["validation"],
        device,
        pad_id=token_to_id["<pad>"]
    )

    print("\nEvaluating test set...")

    test_results = evaluate_model(
        model,
        loaders["test"],
        device,
        pad_id=token_to_id["<pad>"]
    )

    print(
        f"Validation PPL: "
        f"{val_results['perplexity']:.4f}"
    )

    print(
        f"Test Cross-Entropy: "
        f"{test_results['cross_entropy']:.4f}"
    )

    print(
        f"Test PPL: "
        f"{test_results['perplexity']:.4f}"
    )

    print(
        f"Evaluation throughput: "
        f"{test_results['evaluation_tokens_per_second']:.2f} tokens/s"
    )

    # 8. Model size
    checkpoint_path = Path(
        "checkpoints/rnn_best.pt"
    )

    model_size_mb = (
        checkpoint_path.stat().st_size / (1024 ** 2)
    )

    print(f"Model size: {model_size_mb:.2f} MB")

    # 9. Save metrics
    Path("results/metrics").mkdir(
        parents=True,
        exist_ok=True
    )

    metrics = {
        "model": "RNN",
        "vocab_size": len(token_to_id),
        "embedding_dim": config["model"]["embedding_dim"],
        "hidden_dim": config["model"]["hidden_dim"],
        "num_layers": config["model"]["num_layers"],
        "best_epoch": best_epoch,
        "train_loss": history["train_loss"][-1],
        "best_validation_loss": min(history["val_loss"]),
        "validation_perplexity": val_results["perplexity"],
        "test_cross_entropy": test_results["cross_entropy"],
        "test_perplexity": test_results["perplexity"],
        "training_time_seconds": training_time,
        "model_size_mb": model_size_mb,
        "trainable_parameters": parameters,
        "evaluation_time_seconds":
            test_results["evaluation_time_seconds"],
        "evaluation_tokens_per_second":
            test_results["evaluation_tokens_per_second"],
        "evaluated_test_tokens":
            test_results["evaluated_tokens"],
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else "CPU"
        )
    }

    pd.DataFrame([metrics]).to_csv(
        "results/metrics/rnn_results.csv",
        index=False
    )

    pd.DataFrame({
        "epoch": range(1, len(history["train_loss"]) + 1),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"]
    }).to_csv(
        "results/metrics/rnn_history.csv",
        index=False
    )

    # 10. Generate text
    print("\nGenerating sample text...")

    prompts = [
        "the future of technology",
        "in the middle of the night",
        "scientists have discovered"
    ]

    generated_outputs = []

    for prompt in prompts:

        generated = generate_text(
            model=model,
            prompt=prompt,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            device=device,
            max_new_tokens=50
        )

        print("\nPrompt:", prompt)
        print("Generated:", generated)

        generated_outputs.append(
            f"Prompt: {prompt}\n"
            f"Generated: {generated}\n"
        )

    Path("results/generated_text").mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        "results/generated_text/rnn_generations.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(generated_outputs))

    print("\nRNN experiment completed!")


if __name__ == "__main__":
    main()
