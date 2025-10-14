from bpe.fast_token import FastBPETokenizer

from datasets import load_dataset
import json


# def clean_text(text):
#     if not text or not text.strip():
#         return None
#     return text.strip()

webtext = load_dataset("stas/openwebtext-10k", split="train")
dataset = load_dataset("DipamSoni/custom_text_to_sql_dataset", split="train[:10000]")
webcorpus = [txt["text"] for txt in webtext]

def format_for_token(obj):
    return f"{obj['instruction']}\n{obj['input']}\n=> {obj['response']}"

sqlcorpus = [format_for_token(txt) for txt in dataset]

corpus = webcorpus + sqlcorpus

print("Webtext corpus size:", len(webcorpus))
print("SQL corpus size:", len(sqlcorpus))  
print("Total corpus size:", len(corpus))

tokenizer = FastBPETokenizer()
tokenizer.train(corpus, vocab_size=32000)

tokenizer.save("tokenizer_data")