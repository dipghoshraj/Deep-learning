from datasets import load_from_disk
import numpy as np
from tqdm import tqdm

DATASET_PATH = "../tokenized_sql_dataset"
dataset = load_from_disk(DATASET_PATH)

print(f"🚢 Dataset loading........ from {DATASET_PATH}")

tokenized_array = dataset["input_ids"]

# Add tqdm for progress tracking
total_token = sum(len(seq) for seq in tqdm(tokenized_array, desc="🔢 Counting tokens", unit="seqs"))

print(f"🔢 Total tokens: {total_token:,}")

flat_array = np.memmap(
    f"{DATASET_PATH}/flatten_token.memmap", 
    mode='w+', 
    dtype=np.int32, 
    shape=(total_token,)
)
print(f"Created memmap with shape: {flat_array.shape}")

index = 0
for seq in tqdm(tokenized_array, desc="🔥 Flattening", unit="seqs"):
    length = len(seq)
    flat_array[index:index+length] = seq
    index += length

flat_array.flush()
print(f" Flattened memmap saved to: {DATASET_PATH}/flatten_token.memmap")
