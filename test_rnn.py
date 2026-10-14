
import torch
import torch.nn.functional as F

from src.rnn import RNNLanguageModel


def test_rnn():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = RNNLanguageModel(
        vocab_size=100,
        embedding_dim=16,
        hidden_dim=32,
        num_layers=2,
        dropout=0.1
    ).to(device)

    # Batch size = 4, sequence length = 8
    inputs = torch.randint(
        1, 100, (4, 8), device=device
    )

    targets = torch.randint(
        1, 100, (4, 8), device=device
    )

    logits, hidden = model(inputs)

    assert logits.shape == (4, 8, 100)
    assert hidden.shape == (2, 4, 32)

    loss = F.cross_entropy(
        logits.reshape(-1, 100),
        targets.reshape(-1)
    )

    loss.backward()

    assert torch.isfinite(loss)

    assert model.embedding.weight.grad is not None

    # Check that token t does not depend on future tokens.
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

    print("RNN forward pass: PASSED")
    print("RNN backward pass: PASSED")
    print("RNN causal test: PASSED")
    print("Device:", device)
    print("All RNN tests passed!")


if __name__ == "__main__":
    test_rnn()
