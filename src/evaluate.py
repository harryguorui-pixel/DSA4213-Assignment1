
import math
import time

import torch
import torch.nn as nn


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
    pad_id=0
):
    model.eval()

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
        reduction="sum"
    )

    total_loss = 0.0
    total_tokens = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for input_ids, target_ids in dataloader:

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits, _ = model(input_ids)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1)
        )

        total_loss += loss.item()

        total_tokens += (
            target_ids != pad_id
        ).sum().item()

    if device.type == "cuda":
        torch.cuda.synchronize()

    evaluation_time = time.perf_counter() - start_time

    if total_tokens == 0:
        raise ValueError("No valid evaluation tokens.")

    cross_entropy = total_loss / total_tokens
    perplexity = math.exp(cross_entropy)

    throughput = total_tokens / evaluation_time

    return {
        "cross_entropy": cross_entropy,
        "perplexity": perplexity,
        "evaluation_time_seconds": evaluation_time,
        "evaluation_tokens_per_second": throughput,
        "evaluated_tokens": total_tokens
    }


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
