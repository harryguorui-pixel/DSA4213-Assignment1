
import torch
import torch.nn as nn


class TransformerLanguageModel(nn.Module):
    """
    Decoder-only causal Transformer language model.

    Input:
        [batch_size, sequence_length]

    Output:
        logits: [batch_size, sequence_length, vocab_size]
        hidden: None
    """

    def __init__(
        self,
        vocab_size=10000,
        embedding_dim=100,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.2,
        max_seq_length=32,
        pad_id=0
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id
        )

        self.position_embedding = nn.Embedding(
            max_seq_length,
            embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.final_norm = nn.LayerNorm(
            embedding_dim
        )

        self.output_layer = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, input_ids, hidden=None):

        batch_size, seq_length = input_ids.shape

        if seq_length > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds "
                f"max_seq_length {self.max_seq_length}"
            )

        positions = torch.arange(
            seq_length,
            device=input_ids.device
        ).unsqueeze(0)

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        # True means this attention position is blocked.
        causal_mask = torch.triu(
            torch.ones(
                seq_length,
                seq_length,
                device=input_ids.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        x = self.transformer(
            x,
            mask=causal_mask,
            is_causal=True
        )

        x = self.final_norm(x)

        logits = self.output_layer(x)

        return logits, None
