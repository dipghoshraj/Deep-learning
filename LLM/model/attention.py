import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert n_heads * self.head_dim == d_model

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs: torch.Tensor):
        B, seq_length, d_model = inputs.shape

        # Project the input embeddings into Q, K, and V
        Q = (
            self.query(inputs)
            .view(B, seq_length, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        K = (
            self.key(inputs)
            .view(B, seq_length, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        V = (
            self.value(inputs)
            .view(B, seq_length, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )

        # Apply mask to prevent attention to future tokens
        mask = (
            torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
            .bool()
            .to(inputs.device)
        )
        attention_scores = attention_scores.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        # Compute the weighted sum of the values
        attention_output = torch.matmul(self.dropout(attention_weights), V)

        # Concatenate heads and put them back to the original shape
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(B, seq_length, d_model)

        # Apply the final linear transformation
        out = self.fc_out(attention_output)

        return out