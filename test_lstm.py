
import torch
import torch.nn.functional as F

from src.lstm import LSTMLanguageModel


def test_lstm():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = LSTMLanguageModel(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=32,
        num_layers=2,
        dropout=0.1
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

    h_n, c_n = hidden

    assert h_n.shape == (2, 4, 32)
    assert c_n.shape == (2, 4, 32)

    print("LSTM forward pass: PASSED")

    # 2. Backward pass
    loss = F.cross_entropy(
        logits.reshape(-1, 100),
        targets.reshape(-1)
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert model.embedding.weight.grad is not None
    assert model.lstm.weight_ih_l0.grad is not None

    print("LSTM backward pass: PASSED")

    # 3. Causality check
    model.eval()

    with torch.no_grad():

        original, _ = model(inputs)

        modified_inputs = inputs.clone()
        modified_inputs[:, -1] = 1

        modified, _ = model(modified_inputs)

        assert torch.allclose(
            original[:, :-1, :],
            modified[:, :-1, :],
            atol=1e-5
        )

    print("LSTM causal test: PASSED")

    # 4. Check generation with cached hidden states
    with torch.no_grad():

        full_logits, _ = model(inputs)

        prefix_logits, hidden = model(inputs[:, :-1])

        last_logits, _ = model(
            inputs[:, -1:],
            hidden
        )

        assert torch.allclose(
            full_logits[:, -1:, :],
            last_logits,
            atol=1e-5
        )

    print("LSTM hidden-state caching: PASSED")
    print("Device:", device)
    print("All LSTM tests passed!")


if __name__ == "__main__":
    test_lstm()
