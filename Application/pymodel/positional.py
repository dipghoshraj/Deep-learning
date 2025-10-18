import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, context_length, d_model) -> None:
        super().__init__()
        # Create a matrix of shape (context_length, d_model) to store the positional encodings
        pe = torch.zeros(context_length, d_model)

        # Create a vector with positions [0, 1, 2, ..., context_length-1] of shape (context_length, 1)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)

        # Create a vector with the divisor terms based on the dimension
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Compute the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, context_length, d_model)

        # Register pe as a buffer, so it is not considered a parameter but is part of the module's state
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add the positional encodings to the input embeddings
        return x + self.pe[:, : x.size(1), :]