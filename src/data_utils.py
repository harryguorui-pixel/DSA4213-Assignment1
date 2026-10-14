import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    EOS_TOKEN,
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text, lowercase=True):
    """
    Word-level tokenizer.

    Example:
    'The cat sits on the mat.'
    -> ['the', 'cat', 'sits', 'on', 'the', 'mat', '.']
    """

    if lowercase:
        text = text.lower()

    pattern = r"\w+(?:'\w+)*|[^\w\s]"

    return re.findall(pattern, text)


def load_corpus(
    dataset_name="Salesforce/wikitext",
    dataset_config="wikitext-2-raw-v1"
):
    """
    Download WikiText-2 with its original splits.
    """

    dataset = load_dataset(
        dataset_name,
        dataset_config
    )

    return {
        "train": dataset["train"]["text"],
        "validation": dataset["validation"]["text"],
        "test": dataset["test"]["text"],
    }


def tokenize_corpus(texts, lowercase=True):
    """
    Tokenize each non-empty text entry.

    Append EOS to preserve text-entry boundaries.
    """

    tokens = []

    for text in texts:
        if not text.strip():
            continue

        line_tokens = tokenize(
            text,
            lowercase=lowercase
        )

        if line_tokens:
            tokens.extend(line_tokens)
            tokens.append(EOS_TOKEN)

    return tokens


def build_vocab(tokens, vocab_size=10000):
    """
    Build vocabulary using training tokens only.
    """

    counter = Counter(tokens)

    token_to_id = {
        token: idx
        for idx, token in enumerate(SPECIAL_TOKENS)
    }

    for token, count in counter.most_common():
        if token in token_to_id:
            continue

        if len(token_to_id) >= vocab_size:
            break

        token_to_id[token] = len(token_to_id)

    id_to_token = {
        idx: token
        for token, idx in token_to_id.items()
    }

    return token_to_id, id_to_token


def encode_tokens(tokens, token_to_id):
    """
    Convert tokens into integer IDs.
    """

    unk_id = token_to_id[UNK_TOKEN]

    return [
        token_to_id.get(token, unk_id)
        for token in tokens
    ]


def decode_tokens(ids, id_to_token):
    """
    Convert integer IDs back into tokens.
    """

    return [
        id_to_token.get(int(idx), UNK_TOKEN)
        for idx in ids
    ]


class LanguageModelDataset(Dataset):
    """
    Non-overlapping chunks with next-token targets.

    Each sample has:
        input_ids:  [sequence_length]
        target_ids: [sequence_length]
    """

    def __init__(self, token_ids, sequence_length=32):

        self.token_ids = torch.tensor(
            token_ids,
            dtype=torch.long
        )

        self.sequence_length = sequence_length

        self.num_samples = max(
            0,
            (len(self.token_ids) - 1)
            // sequence_length
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        start = idx * self.sequence_length
        end = start + self.sequence_length

        input_ids = self.token_ids[start:end]

        target_ids = self.token_ids[
            start + 1:end + 1
        ]

        return input_ids, target_ids


def prepare_data(config):
    """
    Main preprocessing pipeline.
    """

    data_config = config["data"]
    paths_config = config["paths"]

    output_dir = Path(
        paths_config["processed_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading WikiText-2...")

    corpus = load_corpus(
        data_config["dataset_name"],
        data_config["dataset_config"]
    )

    print("Tokenizing corpus...")

    tokenized = {}

    for split, texts in corpus.items():

        tokenized[split] = tokenize_corpus(
            texts,
            lowercase=data_config["lowercase"]
        )

        print(
            f"{split}: "
            f"{len(tokenized[split]):,} tokens"
        )

    print("Building vocabulary...")

    token_to_id, id_to_token = build_vocab(
        tokenized["train"],
        vocab_size=data_config["vocab_size"]
    )

    print(
        "Vocabulary size:",
        len(token_to_id)
    )

    encoded = {}

    for split, tokens in tokenized.items():

        encoded[split] = encode_tokens(
            tokens,
            token_to_id
        )

        unk_count = sum(
            token == token_to_id[UNK_TOKEN]
            for token in encoded[split]
        )

        unk_rate = (
            unk_count / len(encoded[split])
            if encoded[split]
            else 0
        )

        print(
            f"{split} UNK rate: "
            f"{unk_rate:.2%}"
        )

    with open(
        output_dir / "vocab.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            token_to_id,
            f,
            ensure_ascii=False,
            indent=2
        )

    for split, ids in encoded.items():

        torch.save(
            torch.tensor(ids, dtype=torch.long),
            output_dir / f"{split}.pt"
        )

    print("Preprocessing complete.")

    return encoded, token_to_id, id_to_token


def create_dataloaders(
    encoded,
    sequence_length=32,
    batch_size=64
):
    """
    Create DataLoaders for neural language models.
    """

    loaders = {}

    for split, token_ids in encoded.items():

        dataset = LanguageModelDataset(
            token_ids,
            sequence_length=sequence_length
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

    return loaders