from model.attention import MultiHeadAttention
from model.positional import PositionalEncoding, LearnedPositionalEncoding
import torch.nn.functional as F

import torch.nn as nn
import torch


__all__ = ["MultiHeadAttention", "PositionalEncoding"]

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.att = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, logits):
        att_logits = self.att(logits)
        adn_logits = self.ln1(logits + att_logits)
        logits = self.dropout(adn_logits)
        logits = self.fcn(logits)
        logits = self.ln2(logits + adn_logits)
        return logits


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, context_length):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = PositionalEncoding(
            context_length, d_model
        )
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.linear1 = nn.Linear(d_model, vocab_size)
        self.context_length = context_length

        #parameter sharing
        self.wte.weight = self.linear1.weight

    def forward(self, inputs, targets=None):
        logits = self.wte(inputs)
        logits = self.wpe(logits)
        for block in self.blocks:
            logits = block(logits)
        logits = self.linear1(logits)
        loss = None
        if targets != None:
            batch_size, sequence_length, d_model = logits.shape
            logits = logits.view(batch_size * sequence_length, d_model)
            targets = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        output = inputs.clone()
        for _ in range(max_new_tokens):
            current_seq_length = inputs.size(1)
            if current_seq_length > self.context_length:
                inputs = inputs[:, -self.context_length:]
            logits, _ = self(inputs)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)

            inputs = torch.cat([inputs, idx_next], dim=1)
            output = torch.cat([output, idx_next], dim=1)
        return output



class ModelBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.att = MultiHeadAttention(d_model, n_heads, context_length=1024, dropout=dropout)  # context_length can be passed
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # Pre-LN attention block
        x_norm = self.ln1(x)
        att_out = self.att(x_norm)
        x = x + self.dropout(att_out)

        # Pre-LN feedforward block
        x_norm = self.ln2(x)
        ffn_out = self.fcn(x_norm)
        x = x + self.dropout(ffn_out)
        return x
    

class DecoderGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, context_length, dropout=0.1):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = PositionalEncoding(context_length, d_model)
        self.context_length = context_length

        self.blocks = nn.ModuleList(
            [ModelBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.linear1 = nn.Linear(d_model, vocab_size)

        # Weight tying
        self.linear1.weight = self.wte.weight

    def forward(self, inputs, targets=None):
        logits = self.wte(inputs)        # (B, seq_len, d_model)
        logits = self.wpe(logits)        # Add positional embeddings

        for block in self.blocks:
            logits = block(logits)

        logits = self.linear1(logits)    # (B, seq_len, vocab_size)

        loss = None
        if targets is not None:
            B, seq_len, vocab_size = logits.shape
            logits = logits.view(B * seq_len, vocab_size)
            targets = targets.view(B * seq_len)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, inputs, max_new_tokens, temperature=1.0, top_k=None):
        output = inputs.clone()
        for _ in range(max_new_tokens):
            if output.size(1) > self.context_length:
                inputs = output[:, -self.context_length:]
            else:
                inputs = output

            logits, _ = self(inputs)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
                probs = torch.softmax(top_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(top_indices, -1, next_token)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            output = torch.cat([output, next_token], dim=1)

        return output