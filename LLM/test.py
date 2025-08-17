from bpe.fast_token import FastBPETokenizer
from bpe.byte_encoder import ByteEncoder

from datasets import load_dataset


def clean_text(text):
    if not text or not text.strip():
        return None
    return text.strip()


dataset = load_dataset("stas/openwebtext-10k", split="train")

# Extract just the text column
corpus = [item["text"] for item in dataset]

# ✅ Train tokenizer (tiny vocab for demo, 1k)
tokenizer = FastBPETokenizer()
tokenizer.train(corpus[:2000], vocab_size=1000)

sample = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.tokenize(sample)
print("Tokens:", tokens)

# ✅ Detokenization
detok = "".join(tokens).replace("</w>", " ")
print("Detokenized:", detok)
