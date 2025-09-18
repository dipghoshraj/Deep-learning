from datasets import load_from_disk
import numpy as np

DATASET_PATH = "../tokenized_sql_dataset"
dataset = load_from_disk(DATASET_PATH)


print(f"Dataset loaded from {DATASET_PATH}")

tokenized_array = dataset["input_ids"]
total_token=  sum([len(i) for i in tokenized_array])

print(f"loaded Total tokens: {total_token}")

flat_array = np.memmap(f"{DATASET_PATH}/flatten_token.npy", mode='w+', dtype=np.int32, shape=(total_token,))
print(f"Created memmap at {DATASET_PATH}/flatten_token.npy with shape {flat_array.shape}")

index = 0
for arr in tokenized_array:
    arr = np.asarray(arr, dtype=np.int32)
    length = len(arr)
    flat_array[idx:idx+length] = arr
    idx += length

    if idx % 1_000_00 == 0:
        print(f"Processed {idx} tokens...")


flat_array.flush()
print(f"✅ Flattened memmap saved to: {DATASET_PATH}")