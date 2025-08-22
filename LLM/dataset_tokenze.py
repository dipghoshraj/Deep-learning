
from bpe.fast_token import FastBPETokenizer

from datasets import load_dataset

tokenizer = FastBPETokenizer()
tokenizer.load("tokenizer_data")

ds = load_dataset("DipamSoni/custom_text_to_sql_dataset", split="train")
