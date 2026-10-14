import yaml
import torch

from src.data_utils import (
    set_seed,
    prepare_data,
    create_dataloaders,
    decode_tokens,
)


def main():

    with open(
        "configs/config.yaml",
        "r",
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    set_seed(
        config["project"]["seed"]
    )

    print("=" * 50)
    print("DSA4213 Assignment 1")
    print("=" * 50)

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    encoded, token_to_id, id_to_token = (
        prepare_data(config)
    )

    loaders = create_dataloaders(
        encoded,
        sequence_length=config["data"]["sequence_length"],
        batch_size=config["data"]["batch_size"]
    )

    print("\nDataLoader information:")

    for split, loader in loaders.items():

        print(
            f"{split}: {len(loader)} batches"
        )

    train_loader = loaders["train"]

    input_ids, target_ids = next(
        iter(train_loader)
    )

    print("\nBatch shapes:")

    print(
        "Input:",
        input_ids.shape
    )

    print(
        "Target:",
        target_ids.shape
    )

    print("\nExample input tokens:")

    print(
        decode_tokens(
            input_ids[0][:15].tolist(),
            id_to_token
        )
    )

    print("\nExample target tokens:")

    print(
        decode_tokens(
            target_ids[0][:15].tolist(),
            id_to_token
        )
    )

    assert input_ids.shape == target_ids.shape

    assert input_ids.shape[1] == (
        config["data"]["sequence_length"]
    )

    assert torch.equal(
        input_ids[0, 1:],
        target_ids[0, :-1]
    )

    print("\nAll basic checks passed!")


if __name__ == "__main__":
    main()