
from bpe.fast_token import FastBPETokenizer

from datasets import load_dataset

tokenizer = FastBPETokenizer()
tokenizer.load("tokenizer_data")

ds = load_dataset("DipamSoni/custom_text_to_sql_dataset", split="train")

def tokenize_function(examples):
    texts = [
        f"{instr}\n{inp}\n=> {resp}"
        for instr, inp, resp in zip(
            examples["instruction"], examples["input"], examples["response"]
        )
    ]
    input_ids = [tokenizer.tokenize_to_ids(text) for text in texts]
    return {"input_ids": input_ids}


tokenized_dataset = ds.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=ds.column_names,
    desc="Tokenizing dataset"
)

print(tokenized_dataset[:5])
tokenized_dataset.save_to_disk("tokenized_sql_dataset")