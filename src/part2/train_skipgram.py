
"""
DSA4213 Assignment 1 - Part II
Train self-trained Skip-gram word embeddings.

Uses only the existing WikiText-2 training tokens.
Does not modify Part I checkpoints or processed data.

Run:
    python -m src.part2.train_skipgram
"""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from gensim.models import Word2Vec


CONFIG_PATH = Path("configs/part2.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_training_data(config):
    """
    Load existing Part I vocabulary and encoded training tokens.
    No validation/test data is used.
    """
    data_dir = Path(config["data"]["processed_dir"])

    vocab_path = data_dir / "vocab.json"
    train_path = data_dir / "train.pt"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    train_ids = torch.load(
        train_path,
        map_location="cpu",
        weights_only=True
    ).tolist()

    assert len(vocab) == config["data"]["vocab_size"]

    return vocab, train_ids, vocab_path, train_path


def build_sentences(train_ids, vocab):
    """
    Convert encoded training tokens into Skip-gram sentences.

    - Use EOS as a sentence boundary.
    - Exclude PAD, UNK and EOS from Word2Vec training.
    - Represent each real word by its original token ID.
    - Never create contexts across EOS boundaries.

    Integer IDs are converted to strings for Gensim.
    """
    pad_id = vocab["<pad>"]
    unk_id = vocab["<unk>"]
    eos_id = vocab["<eos>"]

    excluded_ids = {pad_id, unk_id, eos_id}

    sentences = []
    current_sentence = []

    excluded_counts = {
        "<pad>": 0,
        "<unk>": 0,
        "<eos>": 0
    }

    id_to_special = {
        pad_id: "<pad>",
        unk_id: "<unk>",
        eos_id: "<eos>"
    }

    for token_id in train_ids:
        if token_id == eos_id:
            excluded_counts["<eos>"] += 1

            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []

        elif token_id in excluded_ids:
            excluded_counts[id_to_special[token_id]] += 1

        else:
            current_sentence.append(str(token_id))

    if current_sentence:
        sentences.append(current_sentence)

    usable_tokens = sum(len(s) for s in sentences)

    print(f"Training sentences: {len(sentences):,}")
    print(f"Usable training tokens: {usable_tokens:,}")
    print(f"Excluded special tokens: {excluded_counts}")

    assert usable_tokens > 0
    assert len(sentences) > 0

    return sentences, excluded_counts, usable_tokens


def train_skipgram(sentences, config):
    """
    Train Gensim Skip-gram embeddings.
    """
    params = config["embeddings"]["skipgram"]

    print("\nTraining Skip-gram...")
    print(f"Vector size: {params['vector_size']}")
    print(f"Window: {params['window']}")
    print(f"Epochs: {params['epochs']}")
    print(f"Workers: {params['workers']}")
    print(f"Negative samples: {params['negative']}")

    start_time = time.perf_counter()

    model = Word2Vec(
        sentences=sentences,
        vector_size=params["vector_size"],
        window=params["window"],
        min_count=params["min_count"],
        sg=params["sg"],
        workers=params["workers"],
        epochs=params["epochs"],
        negative=params["negative"],
        seed=params["seed"],
        sample=1e-3
    )

    training_time = time.perf_counter() - start_time

    print(f"\nSkip-gram training time: {training_time:.2f}s")
    print(f"Learned vocabulary size: {len(model.wv):,}")

    return model, training_time


def create_embedding_matrix(model, vocab, config):
    """
    Map Gensim embeddings to Part I vocabulary IDs.

    PAD: zero vector
    UNK: seeded random normal vector
    EOS: zero vector
    Other OOV: seeded random normal vector

    Existing Skip-gram vectors are copied unchanged.
    """
    seed = config["project"]["seed"]
    embedding_dim = config["model"]["embedding_dim"]

    rng = np.random.default_rng(seed)

    matrix = rng.normal(
        loc=0.0,
        scale=0.1,
        size=(len(vocab), embedding_dim)
    ).astype(np.float32)

    covered = 0
    missing_words = []

    special_tokens = {"<pad>", "<unk>", "<eos>"}

    for word, token_id in vocab.items():
        if word in special_tokens:
            continue

        key = str(token_id)

        if key in model.wv:
            matrix[token_id] = model.wv[key]
            covered += 1
        else:
            missing_words.append(word)

    matrix[vocab["<pad>"]] = 0.0
    matrix[vocab["<eos>"]] = 0.0

    # UNK keeps its deterministic random initialization.

    assert matrix.shape == (len(vocab), embedding_dim)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()

    eligible_words = len(vocab) - len(special_tokens)

    coverage = covered / eligible_words

    print("\nEmbedding matrix:")
    print(f"Shape: {matrix.shape}")
    print(f"Covered words: {covered:,}/{eligible_words:,}")
    print(f"Coverage: {coverage:.2%}")
    print(f"Missing words: {len(missing_words):,}")

    return matrix, {
        "covered_words": covered,
        "eligible_words": eligible_words,
        "coverage": coverage,
        "missing_words_count": len(missing_words),
        "missing_words_sample": missing_words[:20]
    }


def save_results(
    matrix,
    config,
    training_time,
    coverage_info,
    excluded_counts,
    usable_tokens,
    sentence_count,
    vocab_path,
    train_path
):
    output_dir = Path(config["paths"]["embedding_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / "skipgram_100d.npy"
    metadata_path = output_dir / "skipgram_metadata.json"

    np.save(matrix_path, matrix)

    params = config["embeddings"]["skipgram"]

    metadata = {
        "experiment": "Part II - Self-trained Skip-gram",
        "source": "WikiText-2 training split only",
        "method": "Gensim Word2Vec Skip-gram",
        "seed": config["project"]["seed"],
        "vector_size": params["vector_size"],
        "window": params["window"],
        "min_count": params["min_count"],
        "sg": params["sg"],
        "workers": params["workers"],
        "epochs": params["epochs"],
        "negative": params["negative"],
        "subsampling": 1e-3,
        "training_time_seconds": training_time,
        "sentence_count": sentence_count,
        "usable_training_tokens": usable_tokens,
        "excluded_token_counts": excluded_counts,
        "coverage": coverage_info,
        "special_token_strategy": {
            "<pad>": "zero vector",
            "<unk>": "seeded random normal N(0, 0.1)",
            "<eos>": "zero vector"
        },
        "other_oov_strategy": "seeded random normal N(0, 0.1)",
        "vocab_sha256": sha256_file(vocab_path),
        "train_sha256": sha256_file(train_path),
        "embedding_shape": list(matrix.shape),
        "embedding_dtype": str(matrix.dtype)
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nSaved files:")
    print(f"  {matrix_path}")
    print(f"  {metadata_path}")

    return matrix_path, metadata_path


def verify_saved_matrix(matrix_path, vocab, config):
    print("\nVerifying saved matrix...")

    matrix = np.load(matrix_path, allow_pickle=False)

    assert matrix.shape == (
        len(vocab),
        config["model"]["embedding_dim"]
    )

    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()

    assert np.all(matrix[vocab["<pad>"]] == 0)
    assert np.all(matrix[vocab["<eos>"]] == 0)

    assert np.any(matrix[vocab["<unk>"]] != 0)

    print("Matrix verification: PASSED")


def main():
    print("=" * 65)
    print("DSA4213 PART II - SELF-TRAINED SKIP-GRAM")
    print("=" * 65)

    config = load_config()

    set_seed(config["project"]["seed"])

    vocab, train_ids, vocab_path, train_path = (
        load_training_data(config)
    )

    print(f"Vocabulary size: {len(vocab):,}")
    print(f"Training tokens: {len(train_ids):,}")

    sentences, excluded_counts, usable_tokens = (
        build_sentences(train_ids, vocab)
    )

    model, training_time = train_skipgram(
        sentences,
        config
    )

    matrix, coverage_info = create_embedding_matrix(
        model,
        vocab,
        config
    )

    matrix_path, metadata_path = save_results(
        matrix=matrix,
        config=config,
        training_time=training_time,
        coverage_info=coverage_info,
        excluded_counts=excluded_counts,
        usable_tokens=usable_tokens,
        sentence_count=len(sentences),
        vocab_path=vocab_path,
        train_path=train_path
    )

    verify_saved_matrix(
        matrix_path,
        vocab,
        config
    )

    print("\n" + "=" * 65)
    print("SKIP-GRAM TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print(f"Training time: {training_time:.2f}s")
    print(f"Coverage: {coverage_info['coverage']:.2%}")


if __name__ == "__main__":
    main()
