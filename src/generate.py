
import torch

from src.data_utils import (
    tokenize,
    encode_tokens,
    decode_tokens
)


@torch.no_grad()
def generate_text(
    model,
    prompt,
    token_to_id,
    id_to_token,
    device,
    max_new_tokens=50
):
    model.eval()

    prompt_tokens = tokenize(prompt)

    prompt_ids = encode_tokens(
        prompt_tokens,
        token_to_id
    )

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    generated_ids = prompt_ids.copy()

    # Process the complete prompt once.
    logits, hidden = model(input_ids)

    eos_id = token_to_id["<eos>"]
    pad_id = token_to_id["<pad>"]

    for _ in range(max_new_tokens):

        next_logits = logits[0, -1, :].clone()

        # PAD is not a valid generated token.
        next_logits[pad_id] = float("-inf")

        next_token = int(
            torch.argmax(next_logits).item()
        )

        generated_ids.append(next_token)

        if next_token == eos_id:
            break

        next_input = torch.tensor(
            [[next_token]],
            dtype=torch.long,
            device=device
        )

        logits, hidden = model(
            next_input,
            hidden
        )

    generated_tokens = decode_tokens(
        generated_ids,
        id_to_token
    )

    return " ".join(generated_tokens)



@torch.no_grad()
def generate_transformer_text(
    model,
    prompt,
    token_to_id,
    id_to_token,
    device,
    max_new_tokens=50
):
    """
    Greedy decoding for causal Transformer.

    Uses a sliding context window.
    """

    model.eval()

    prompt_tokens = tokenize(prompt)

    generated_ids = encode_tokens(
        prompt_tokens,
        token_to_id
    )

    eos_id = token_to_id["<eos>"]
    pad_id = token_to_id["<pad>"]

    if not generated_ids:
        raise ValueError("Prompt must contain at least one token.")

    for _ in range(max_new_tokens):

        context_ids = generated_ids[
            -model.max_seq_length:
        ]

        input_ids = torch.tensor(
            [context_ids],
            dtype=torch.long,
            device=device
        )

        logits, _ = model(input_ids)

        next_logits = logits[0, -1, :].clone()

        next_logits[pad_id] = float("-inf")

        next_token = int(
            torch.argmax(next_logits).item()
        )

        generated_ids.append(next_token)

        if next_token == eos_id:
            break

    generated_tokens = decode_tokens(
        generated_ids,
        id_to_token
    )

    return " ".join(generated_tokens)
