# src/models/textcnn.py
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        num_classes: int,
        filter_sizes: List[int],
        num_filters: int,
        pad_idx: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_idx,
        )

        # conv input [B, E, L]，output [B, num_filters, L']
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=emb_dim,
                    out_channels=num_filters,
                    kernel_size=fs,
                )
                for fs in filter_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L]  -> logits: [B, num_classes]
        """
        # [B, L] -> [B, L, E]
        emb = self.embedding(x)
        # [B, E, L]
        emb = emb.transpose(1, 2) 

        conv_outputs = []
        for conv in self.convs:
            # conv_out: [B, num_filters, L']
            c = conv(emb)
            c = F.relu(c)
            # global max pooling over time
            # pooled: [B, num_filters]
            pooled = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        # [B, num_filters * len(filter_sizes)]
        h = torch.cat(conv_outputs, dim=1)

        h = self.dropout(h)
        logits = self.fc(h)
        return logits
