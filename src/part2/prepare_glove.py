
"""
DSA4213 Assignment 1 - Part II
Prepare public pretrained GloVe 100d embeddings.

Run:
    python -m src.part2.prepare_glove
"""

import hashlib
import json
import time
from pathlib import Path

import gensim
import gensim.downloader as api
import numpy as np
import yaml


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


def load_vocab(config):
    path = Path(config["data"]["processed_dir"]) / "vocab.json"

    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    assert len(vocab) == config["data"]["vocab_size"]
    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert vocab["<eos>"] == 2

    return vocab, path


def load_glove(config):
    name = config["embeddings"]["glove"]["name"]

    print(f"\nLoading public pretrained embeddings: {name}")
    print("The first run may download the model.")

    start = time.perf_counter()

    glove = api.load(name)

    elapsed = time.perf_counter() - start

    expected_dim = config["model"]["embedding_dim"]

    assert glove.vector_size == expected_dim

    print(f"Loaded vectors: {len(glove):,}")
    print(f"Vector dimension: {glove.vector_size}")
    print(f"Load/download time: {elapsed:.2f}s")

    return glove, elapsed


def build_embedding_matrix(glove, vocab, config):
    dim = config["model"]["embedding_dim"]
    seed = config["embeddings"]["glove"]["oov_seed"]

    rng = np.random.default_rng(seed)

    matrix = rng.normal(
        loc=0.0,
        scale=0.1,
        size=(len(vocab), dim)
    ).astype(np.float32)

    special_tokens = {"<pad>", "<unk>", "<eos>"}

    covered = 0
    missing_words = []

    for word, token_id in vocab.items():
        if word in special_tokens:
            continue

        if word in glove:
            matrix[token_id] = glove[word]
            covered += 1
        else:
            missing_words.append(word)

    matrix[vocab["<pad>"]] = 0.0
    matrix[vocab["<eos>"]] = 0.0

    eligible = len(vocab) - len(special_tokens)
    oov_count = len(missing_words)

    coverage = covered / eligible

    assert covered + oov_count == eligible
    assert matrix.shape == (len(vocab), dim)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()

    print("\nGloVe vocabulary coverage:")
    print(f"Covered words: {covered:,}")
    print(f"OOV words: {oov_count:,}")
    print(f"Eligible words: {eligible:,}")
    print(f"Coverage: {coverage:.2%}")
    print(f"OOV rate: {1 - coverage:.2%}")
    print(f"OOV examples: {missing_words[:20]}")

    return matrix, {
        "covered_words": covered,
        "oov_words": oov_count,
        "eligible_words": eligible,
        "coverage": coverage,
        "oov_rate": 1 - coverage,
        "oov_examples": missing_words[:30]
    }


def save_results(
    matrix,
    coverage_info,
    config,
    vocab_path,
    load_time
):
    output_dir = Path(config["paths"]["embedding_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / "glove_100d.npy"
    metadata_path = output_dir / "glove_metadata.json"

    np.save(matrix_path, matrix)

    metadata = {
        "experiment": "Part II - Public Pretrained GloVe",
        "embedding_name": config["embeddings"]["glove"]["name"],
        "source": "Public GloVe Wikipedia + Gigaword",
        "library": "gensim.downloader",
        "gensim_version": gensim.__version__,
        "embedding_dimension": config["model"]["embedding_dim"],
        "vocab_size": len(matrix),
        "coverage": coverage_info,
        "oov_initialization": {
            "strategy": "random_normal",
            "mean": 0.0,
            "std": 0.1,
            "seed": config["embeddings"]["glove"]["oov_seed"]
        },
        "special_tokens": {
            "<pad>": "zero vector",
            "<unk>": "seeded random normal N(0, 0.1)",
            "<eos>": "zero vector"
        },
        "load_download_time_seconds": load_time,
        "vocab_sha256": sha256_file(vocab_path),
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": str(matrix.dtype),
        "uses_external_corpus": True
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    return matrix_path, metadata_path


def verify_results(matrix_path, vocab, config):
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

    print("\nMatrix verification: PASSED")


def main():
    print("=" * 65)
    print("DSA4213 PART II - PUBLIC PRETRAINED GLOVE")
    print("=" * 65)

    config = load_config()
    vocab, vocab_path = load_vocab(config)

    print(f"Part I vocabulary size: {len(vocab):,}")

    glove, load_time = load_glove(config)

    matrix, coverage_info = build_embedding_matrix(
        glove,
        vocab,
        config
    )

    matrix_path, metadata_path = save_results(
        matrix,
        coverage_info,
        config,
        vocab_path,
        load_time
    )

    verify_results(matrix_path, vocab, config)

    print("\nSaved files:")
    print(f"  {matrix_path}")
    print(f"  {metadata_path}")

    print("\n" + "=" * 65)
    print("GLOVE PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print(f"Coverage: {coverage_info['coverage']:.2%}")
    print(f"OOV words: {coverage_info['oov_words']:,}")


if __name__ == "__main__":
    main()
