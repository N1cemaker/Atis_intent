# src/models/bilstm.py
import torch
import torch.nn as nn


class BiLSTMIntent(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_classes: int,
        pad_idx: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,  # PyTorch 要求 num_layers>1 才能用
        )

        num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L] -> logits: [B, num_classes]
        """
        # [B, L] -> [B, L, E]
        emb = self.embedding(x)

        # output: [B, L, H * num_directions]
        # h_n: [num_layers * num_directions, B, H]
        output, (h_n, c_n) = self.lstm(emb)
        # h_n shape: [num_layers * num_directions, B, H]
        num_directions = 2 if self.bidirectional else 1
        h_n = h_n.view(self.num_layers, num_directions, x.size(0), self.hidden_dim)
        # last layer: [num_directions, B, H]
        last_layer_h = h_n[-1]

        if self.bidirectional:
            # [B, 2H]
            h_cat = torch.cat([last_layer_h[0], last_layer_h[1]], dim=1)
        else:
            # [B, H]
            h_cat = last_layer_h[0]

        h_cat = self.dropout(h_cat)
        logits = self.fc(h_cat)
        return logits
