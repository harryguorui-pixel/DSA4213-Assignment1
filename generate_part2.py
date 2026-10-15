
"""
DSA4213 Assignment 1 - Part II
Generate text using three RNN embedding variants.

Usage:
    python generate_part2.py

Uses the same prompts and greedy decoding as Part I.
Does not retrain models or overwrite Part I outputs.
"""

import json
from pathlib import Path

import torch
import yaml

from src.rnn import RNNLanguageModel
from src.generate import generate_text


PROMPTS = [
    "the future of technology",
    "in the middle of the night",
    "scientists have discovered"
]

MAX_NEW_TOKENS = 50

CHECKPOINTS = {
    "A": {
        "name": "Random Trainable",
        "path": "checkpoints/rnn_best.pt"
    },
    "B": {
        "name": "Skip-gram Frozen",
        "path": "checkpoints/part2/rnn_skipgram_best.pt"
    },
    "C": {
        "name": "GloVe Frozen",
        "path": "checkpoints/part2/rnn_glove_best.pt"
    }
}


def main():
    with open(
        "configs/part2.yaml",
        "r",
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data"]["processed_dir"])

    with open(
        data_dir / "vocab.json",
        "r",
        encoding="utf-8"
    ) as f:
        vocab = json.load(f)

    id_to_token = {
        int(idx): word
        for word, idx in vocab.items()
    }

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(config["paths"]["generated_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "part2_generated_text.json"

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    torch.manual_seed(config["project"]["seed"])

    print("=" * 65)
    print("PART II - REPRESENTATIVE TEXT GENERATION")
    print("=" * 65)
    print(f"Device: {device}")
    print("Decoding: Greedy")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")

    results = {}

    for setting, info in CHECKPOINTS.items():
        print("\n" + "-" * 65)
        print(f"Setting {setting}: {info['name']}")
        print("-" * 65)

        params = config["model"]

        model = RNNLanguageModel(
            vocab_size=len(vocab),
            embedding_dim=params["embedding_dim"],
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            pad_id=vocab["<pad>"]
        )

        checkpoint = torch.load(
            info["path"],
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        model = model.to(device)
        model.eval()

        results[setting] = {
            "embedding": info["name"],
            "checkpoint": info["path"],
            "generations": []
        }

        for prompt in PROMPTS:
            generated = generate_text(
                model=model,
                prompt=prompt,
                token_to_id=vocab,
                id_to_token=id_to_token,
                device=device,
                max_new_tokens=MAX_NEW_TOKENS
            )

            results[setting]["generations"].append({
                "prompt": prompt,
                "generated_text": generated
            })

            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated}")

        del model

    output = {
        "decoding": "greedy",
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompts": PROMPTS,
        "results": results
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    assert len(results) == 3

    for setting in results:
        assert len(results[setting]["generations"]) == 3

    print("\n" + "=" * 65)
    print("TEXT GENERATION COMPLETED")
    print("=" * 65)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
