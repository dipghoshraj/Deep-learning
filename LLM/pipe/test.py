# from bpe.fast_token import FastBPETokenizer
# tokenizer = FastBPETokenizer()

# tokenizer.load("../tokenizer_data")
# tokens = tokenizer.tokenize_to_ids("find")


# text = ''''Which tests have 'Pass' results? Return the dates when the tests were taken, and count them by a line chart, and I want to display by the X-axis in asc.
# CREATE TABLE Courses (course_id INTEGER,author_id INTEGER,subject_id INTEGER,course_name VARCHAR(120),course_description VARCHAR(255))
# =>
# '''


# # print(tokenizer.tokenize_to_ids(text))

# print(tokenizer.decode_from_ids(ids_decode))

import torch
from torch.cuda.amp import autocast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cpu"

if torch.cuda.is_available():
  device = "cuda"

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast(
    tokenizer_file="./slm/tokenizer.json",
    truncation=True,
    unk_token="<UNK>",
    pad_token="<PAD>",
    bos_token="<BOS>",
    eos_token="<EOS>",
    max_length=50
)
import math

class PositionalEncoding(nn.Module):
    def __init__(self, context_length, d_model):
        super().__init__()
        self.register_buffer("pe", self._build_pe(context_length, d_model))
        self.dropout = nn.Dropout(0.1)



    def _build_pe(self, length, d_model):
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: [1, length, d_model]

    def forward(self, x):
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.fc_out(self.dropout(attn_output))
    
class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, context_length, dropout=0.1):
        super().__init__()
        self.context_length = context_length
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = PositionalEncoding(context_length, d_model)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # Weight tying

        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)


    def forward(self, x, targets=None):
        B, T = x.size()
        assert T <= self.context_length, f"Input length {T} exceeds context length {self.context_length}"

        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        x = self.wte(x)
        x = self.wpe(x)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Trim targets/logits to same length
            min_len = min(logits.size(1), targets.size(1))
            logits = logits[:, :min_len, :]
            targets = targets[:, :min_len]

            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                label_smoothing=0.1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        top_k=50,
        top_p=0.9,
        temperature=1.0,
        repetition_penalty=1.1
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # get logits for last token
            logits = logits / temperature

            # Repetition penalty
            for i in range(idx.size(0)):
                for token in set(idx[i].tolist()):
                    if token < logits.size(-1):
                        logits[i, token] /= repetition_penalty

            # Top-k
            if top_k is not None and top_k > 0:
                topk_values, _ = torch.topk(logits, top_k)
                min_vals = topk_values[:, -1].unsqueeze(1)
                logits = torch.where(logits < min_vals, float('-inf'), logits)

            # Top-p (nucleus)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)

                sorted_mask = cum_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = 0

                masked_sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
                logits = logits.clone()  # clone before in-place scatter_
                logits.scatter_(1, sorted_indices, masked_sorted_logits)


            # Convert to probabilities safely
            probs = F.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

            probs_sum = probs.sum(dim=-1, keepdim=True)
            probs_sum[probs_sum == 0] = 1  # avoid divide-by-zero
            probs = probs / probs_sum

            # Sample
            try:
                next_token = torch.multinomial(probs, num_samples=1)
            except RuntimeError as e:
                print("Error in multinomial sampling.")
                print("Probs:", probs)
                raise e

            idx = torch.cat((idx, next_token), dim=1)

        return idx



basic_model = GPT(vocab_size=8000, d_model=256, n_heads=8, n_layers=6, context_length=126).to(device)
basic_model.to(device)

checkpoint = torch.load("./LLM/checkpoints/slm_epoch_10.pt", map_location=device)
basic_model.load_state_dict(checkpoint["model_state_dict"])
basic_model.eval()


prompt = "Filling"

inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
input_ids = inputs["input_ids"].to(device)

# Optional: check if input tokens are valid
print("Max token ID:", input_ids.max().item())
print("Vocab size:", tokenizer.vocab_size)


# --- Generate ---
with torch.no_grad():
    output_ids = basic_model.generate(input_ids, max_new_tokens=80)

# Decode
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated output:", output_text)
