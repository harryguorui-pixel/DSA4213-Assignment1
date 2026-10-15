import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


SETTINGS = ["random", "skipgram", "glove"]

CLASS_NAMES = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech",
]

LM_PPL = {
    "random": 102.4333,
    "skipgram": 95.3216,
    "glove": 96.9657,
}


def load_config():
    with open(
        "configs/part3.yaml",
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def evaluate_setting(setting, config, device):
    feature_path = (
        Path("data/part3/features")
        / f"{setting}_test.pt"
    )

    data = torch.load(
        feature_path,
        map_location="cpu",
        weights_only=True,
    )

    features = data["features"].float()
    labels = data["labels"].long()

    model = nn.Linear(
        config["model"]["hidden_dim"],
        config["model"]["num_classes"],
    ).to(device)

    checkpoint_path = (
        Path(config["paths"]["checkpoint_dir"])
        / f"classifier_{setting}_best.pt"
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

    model.eval()

    with torch.no_grad():
        logits = model(features.to(device))
        probabilities = torch.softmax(logits, dim=1)

        predictions = probabilities.argmax(dim=1).cpu().numpy()
        confidence = probabilities.max(dim=1).values.cpu().numpy()

    y_true = labels.numpy()

    accuracy = accuracy_score(y_true, predictions)
    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=list(range(4)),
    )

    return {
        "setting": setting,
        "best_epoch": checkpoint["epoch"],
        "lm_ppl": LM_PPL[setting],
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "y_true": y_true,
        "y_pred": predictions,
        "confidence": confidence,
        "confusion_matrix": cm,
    }


def save_confusion_matrix(result, figure_dir):
    cm = result["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")

    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))

    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(
        f"{result['setting'].upper()} - Confusion Matrix"
    )

    for i in range(4):
        for j in range(4):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    plt.tight_layout()
    plt.savefig(
        figure_dir / f"{result['setting']}_confusion_matrix.png",
        dpi=200,
    )
    plt.close()


def save_errors(result, texts, metrics_dir):
    if len(texts) != len(result["y_true"]):
        raise ValueError("Test texts and labels are not aligned.")

    errors = []

    for idx, (true, pred, conf) in enumerate(
        zip(
            result["y_true"],
            result["y_pred"],
            result["confidence"],
        )
    ):
        if true != pred:
            errors.append({
                "index": idx,
                "true_label": CLASS_NAMES[int(true)],
                "predicted_label": CLASS_NAMES[int(pred)],
                "confidence": float(conf),
                "text": texts[idx],
            })

    # Select representative high-confidence mistakes.
    errors.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    output_path = (
        metrics_dir
        / f"{result['setting']}_representative_errors.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "true_label",
                "predicted_label",
                "confidence",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(errors[:20])

    return len(errors)


def main():
    config = load_config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    metrics_dir = Path(config["paths"]["metrics_dir"])
    figure_dir = Path(config["paths"]["figure_dir"])

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    texts_path = (
        Path(config["paths"]["processed_dir"])
        / "test_texts.json"
    )

    with open(texts_path, "r", encoding="utf-8") as f:
        test_texts = json.load(f)

    print("=" * 65)
    print("PART III - UNIFIED DOWNSTREAM EVALUATION")
    print("=" * 65)
    print("Device:", device)

    summary = []

    reference_labels = None

    for setting in SETTINGS:
        result = evaluate_setting(
            setting, config, device
        )

        if reference_labels is None:
            reference_labels = result["y_true"]
        else:
            assert np.array_equal(
                reference_labels,
                result["y_true"],
            ), "Test labels differ across settings."

        save_confusion_matrix(result, figure_dir)

        num_errors = save_errors(
            result,
            test_texts,
            metrics_dir,
        )

        report = classification_report(
            result["y_true"],
            result["y_pred"],
            labels=list(range(4)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )

        with open(
            metrics_dir / f"{setting}_classification_report.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report)

        print(f"\nSetting: {setting.upper()}")
        print(f"Best Epoch: {result['best_epoch']}")
        print(f"LM Test PPL: {result['lm_ppl']:.4f}")
        print(f"Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"Test Macro-F1: {result['test_macro_f1']:.4f}")
        print(f"Misclassified Samples: {num_errors}")
        print("\n", report)

        summary.append({
            "setting": setting,
            "lm_test_ppl": result["lm_ppl"],
            "classifier_best_epoch": result["best_epoch"],
            "test_accuracy": result["test_accuracy"],
            "test_macro_f1": result["test_macro_f1"],
            "misclassified_samples": num_errors,
        })

    with open(
        metrics_dir / "part3_final_comparison.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "lm_test_ppl",
                "classifier_best_epoch",
                "test_accuracy",
                "test_macro_f1",
                "misclassified_samples",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    for row in summary:
        print(
            f"{row['setting']:10s} | "
            f"PPL: {row['lm_test_ppl']:8.4f} | "
            f"Accuracy: {row['test_accuracy']:.4f} | "
            f"Macro-F1: {row['test_macro_f1']:.4f}"
        )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()