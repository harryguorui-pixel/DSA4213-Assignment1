
import torch
import torch.nn.functional as F

from src.transformer import TransformerLanguageModel


def test_transformer():

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = TransformerLanguageModel(
        vocab_size=100,
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        feedforward_dim=64,
        dropout=0.0,
        max_seq_length=16,
        pad_id=0
    ).to(device)

    inputs = torch.randint(
        1, 100, (4, 8), device=device
    )

    targets = torch.randint(
        1, 100, (4, 8), device=device
    )

    # 1. Forward pass
    logits, hidden = model(inputs)

    assert logits.shape == (4, 8, 100)
    assert hidden is None

    print("Transformer forward pass: PASSED")

    # 2. Backward pass
    loss = F.cross_entropy(
        logits.reshape(-1, 100),
        targets.reshape(-1)
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert model.token_embedding.weight.grad is not None

    print("Transformer backward pass: PASSED")

    # 3. Causal masking test
    model.eval()

    with torch.no_grad():

        original, _ = model(inputs)

        modified_inputs = inputs.clone()

        # Change the last token.
        modified_inputs[:, -1] = (
            modified_inputs[:, -1] % 99
        ) + 1

        modified, _ = model(modified_inputs)

        # Earlier positions must not change.
        assert torch.allclose(
            original[:, :-1, :],
            modified[:, :-1, :],
            atol=1e-4,
            rtol=1e-4
        )

    print("Transformer causal masking: PASSED")

    # 4. Context length test
    too_long = torch.randint(
        1, 100, (1, 17), device=device
    )

    try:
        model(too_long)
        raise AssertionError(
            "Expected a sequence-length error."
        )
    except ValueError:
        pass

    print("Transformer context length: PASSED")

    print("Device:", device)
    print("All Transformer tests passed!")


if __name__ == "__main__":
    test_transformer()
