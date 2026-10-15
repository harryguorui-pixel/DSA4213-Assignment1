import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset

from src.data_utils import (
    tokenize,
    encode_tokens,
    PAD_TOKEN,
    UNK_TOKEN,
    set_seed,
)


def stratified_sample(dataset, samples_per_class, seed):
    """
    Select an equal number of samples from each class.
    """
    rng = random.Random(seed)

    groups = {}

    for idx, label in enumerate(dataset["label"]):
        groups.setdefault(int(label), []).append(idx)

    selected = []

    for label in sorted(groups):
        indices = groups[label]

        if len(indices) < samples_per_class:
            raise ValueError(
                f"Not enough samples for class {label}"
            )

        selected.extend(
            rng.sample(indices, samples_per_class)
        )

    rng.shuffle(selected)

    return dataset.select(selected)


def encode_dataset(dataset, token_to_id, max_length):
    """
    Convert AG News texts into padded token ID sequences.
    """
    pad_id = token_to_id[PAD_TOKEN]
    unk_id = token_to_id[UNK_TOKEN]

    all_input_ids = []
    all_labels = []
    all_lengths = []

    total_tokens = 0
    unknown_tokens = 0
    truncated_samples = 0

    for example in dataset:
        tokens = tokenize(example["text"], lowercase=True)

        if len(tokens) > max_length:
            truncated_samples += 1

        tokens = tokens[:max_length]

        ids = encode_tokens(tokens, token_to_id)

        if not ids:
            ids = [unk_id]

        length = len(ids)

        unknown_tokens += sum(
            token_id == unk_id for token_id in ids
        )
        total_tokens += length

        padded_ids = ids + [pad_id] * (max_length - length)

        all_input_ids.append(padded_ids)
        all_labels.append(int(example["label"]))
        all_lengths.append(length)

    return {
        "input_ids": torch.tensor(
            all_input_ids, dtype=torch.long
        ),
        "labels": torch.tensor(
            all_labels, dtype=torch.long
        ),
        "lengths": torch.tensor(
            all_lengths, dtype=torch.long
        ),
        "statistics": {
            "samples": len(all_labels),
            "total_tokens": total_tokens,
            "unknown_tokens": unknown_tokens,
            "unk_rate": unknown_tokens / max(total_tokens, 1),
            "truncated_samples": truncated_samples,
            "truncation_rate": (
                truncated_samples / max(len(all_labels), 1)
            ),
        },
    }


def main():
    config_path = Path("configs/part3.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["project"]["seed"]
    set_seed(seed)

    data_config = config["data"]
    paths = config["paths"]

    output_dir = Path(paths["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        paths["vocab_path"], "r", encoding="utf-8"
    ) as f:
        token_to_id = json.load(f)

    print("=" * 60)
    print("PART III - AG NEWS DATA PREPARATION")
    print("=" * 60)

    print("\nLoading AG News...")
    dataset = load_dataset(data_config["dataset_name"])

    train_size = data_config["train_samples"]
    val_size = data_config["val_samples"]

    if train_size % 4 != 0 or val_size % 4 != 0:
        raise ValueError(
            "Train and validation sizes must be divisible by 4."
        )

    # Sample disjoint subsets from the official training split.
    sampled = stratified_sample(
        dataset["train"],
        (train_size + val_size) // 4,
        seed,
    )

    # Split each class independently to preserve class balance.
    rng = random.Random(seed + 1)

    train_indices = []
    val_indices = []

    for label in range(4):
        class_indices = [
            idx
            for idx, value in enumerate(sampled["label"])
            if int(value) == label
        ]

        rng.shuffle(class_indices)

        n_train = train_size // 4

        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    splits = {
        "train": sampled.select(train_indices),
        "validation": sampled.select(val_indices),
        "test": dataset["test"],
    }

    print("\nEncoding datasets...")

    metadata = {
        "dataset": data_config["dataset_name"],
        "seed": seed,
        "max_length": data_config["max_length"],
        "vocab_size": len(token_to_id),
        "splits": {},
    }

    for split_name, split_dataset in splits.items():
        encoded = encode_dataset(
            split_dataset,
            token_to_id,
            data_config["max_length"],
        )

        save_data = {
            "input_ids": encoded["input_ids"],
            "labels": encoded["labels"],
            "lengths": encoded["lengths"],
        }

        torch.save(
            save_data,
            output_dir / f"{split_name}.pt",
        )

        # Save original texts for later error analysis.
        with open(
            output_dir / f"{split_name}_texts.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                list(split_dataset["text"]),
                f,
                ensure_ascii=False,
                indent=2,
            )

        stats = encoded["statistics"]

        metadata["splits"][split_name] = {
            **stats,
            "class_counts": dict(
                Counter(int(x) for x in split_dataset["label"])
            ),
        }

        print(f"\n{split_name.upper()}")
        print(f"Samples: {stats['samples']}")
        print(f"UNK rate: {stats['unk_rate']:.2%}")
        print(
            f"Truncation rate: "
            f"{stats['truncation_rate']:.2%}"
        )
        print(
            "Class distribution:",
            metadata["splits"][split_name]["class_counts"],
        )

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\nData preparation completed successfully.")
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()