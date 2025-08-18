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

#Train tokenizer (tiny vocab for demo, 1k)
tokenizer = FastBPETokenizer()
tokenizer.train(corpus[:2000], vocab_size=1000)

sample2=  'what will the population of Asia be when Latin America/Caribbean is 783 (7.5%)?\nCREATE TABLE table_22767 ("Year" real,"World" real,"Asia" text,"Africa" text,"Europe" text,"Latin America/Caribbean" text,"Northern America" text,"Oceania" text)\n=> SELECT "Asia" FROM table_22767 WHERE "Latin America/Caribbean" = \'783 (7.5%)\''
tokens = tokenizer.tokenize(sample2)
print("Tokens:", tokens)

# Detokenization
detok = "".join(tokens).replace("</w>", " ")
print("Detokenized:", detok)


# Token IDs
token_ids = tokenizer.tokenize_to_ids(sample2)
print("Token IDs:", token_ids)

# Decode from IDs
decoded_text = tokenizer.decode_from_ids(token_ids)
print("Decoded Text:", decoded_text)
