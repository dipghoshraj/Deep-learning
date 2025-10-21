import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, context_length: int, dropout=0.1):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert n_heads * self.head_dim == d_model, "d_model must be divisible by n_heads"

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Precompute causal mask for context length and register buffer
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, inputs: torch.Tensor):
        B, seq_length, d_model = inputs.shape

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

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Use only mask for current sequence length
        mask = self.mask[:seq_length, :seq_length]
        attention_scores = attention_scores.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(B, seq_length, d_model)

        out = self.fc_out(attention_output)
        return out