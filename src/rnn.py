
import torch
import torch.nn as nn


class RNNLanguageModel(nn.Module):
    """
    Vanilla RNN for next-token prediction.

    Input:
        [batch_size, sequence_length]

    Output:
        [batch_size, sequence_length, vocab_size]
    """

    def __init__(
        self,
        vocab_size=10000,
        embedding_dim=100,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
        pad_id=0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, input_ids, hidden=None):

        embedded = self.embedding(input_ids)

        rnn_output, hidden = self.rnn(
            embedded,
            hidden
        )

        logits = self.output_layer(rnn_output)

        return logits, hidden
