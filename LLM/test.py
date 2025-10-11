# from bpe.fast_token import FastBPETokenizer
# tokenizer = FastBPETokenizer()

# tokenizer.load("../tokenizer_data")
# tokens = tokenizer.tokenize_to_ids("find")


# text = ''''Which tests have 'Pass' results? Return the dates when the tests were taken, and count them by a line chart, and I want to display by the X-axis in asc.
# CREATE TABLE Courses (course_id INTEGER,author_id INTEGER,subject_id INTEGER,course_name VARCHAR(120),course_description VARCHAR(255))
# =>
# '''


# # print(tokenizer.tokenize_to_ids(text))
# ids_decode = [37,7,81,165,171,166,256,252,247,168,7,68,109,244,7,36,240,253,191,254,244,43,36,70,136,254,277,202,264,132,113,247,286,146,264,256,252,247,283,151,254,98,145,16,105,125,225,255,263,199,124,93,191,178,138,129,254,16,105,60,283,103,255,270,132,184,234,119,124,264,82,17,92,289,185,179,109,125,19,52,76,50,226,244,154,9,125,226,249,90,59,66,73,54,57,54,70,15,92,273,260,220,90,59,66,73,54,57,54,70,15,253,120,188,140,254,90,59,66,73,54,57,54,70,15,125,226,249,89,201,102,80,45,70,50,58,45,70,9,22,24,10,15,125,226,249,89,132,153,125,242,230,269,80,45,70,50,58,45,70,9,23,27,27,10,11,41,42,36,38,125,254,145,208,212,254,1593,1593,1593,1593,1593,1593,1593,132,89,125,211,138,136,90,182,15,238,101,136,89,258,12,76,50,58,45,70,15,238,273,164,247,9,1,50,58,54,70,9,92,252,45,59,66,73,70,15,265,125,242,1350,1350,1350,1350,1350,1350,1350,1350,1350,1350,1350,1350,1350,1350,1350,1350,1350,1350,1350,1350,1350,1350,1350,254,273,136,18,125,242,230,107,170,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,1014,198,2240,2240,2240,2240,2240,225,225,255,83,230,269,9,240,89,157,191,1983,1983,1983,1983,99,89,201,102,80,45,70,9,125,191,94,36,80,1593,1593]

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


import math

class PositionalEncoding(nn.Module):
    def __init__(self, context_length, d_model):
        super().__init__()
        self.register_buffer("pe", self._build_pe(context_length, d_model))

    def _build_pe(self, length, d_model):
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: [1, length, d_model]

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


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
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
    
basic_model = GPT(vocab_size=8000, d_model=256, n_heads=8, n_layers=6, context_length=50).to(device)
basic_model.to(device)



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

checkpoint = torch.load("./LLM/checkpoints/slm_epoch_5.pt", map_location=device)
basic_model.load_state_dict(checkpoint["model_state_dict"])
basic_model.eval()


prompt = "I've been feeling so sad and overwhelmed"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

print("Input IDs:", input_ids)

# --- Generate ---
with torch.no_grad():
    output_ids = basic_model.generate(input_ids, max_new_tokens=80)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Model output:", output_text)
