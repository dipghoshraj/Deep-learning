import numpy as np
flattened = np.memmap(
    "../tokenized_sql_dataset/flatten_token.memmap", 
    dtype=np.int32, 
    mode="r"
)


print(f"Loaded flattened tokens with shape: {flattened.shape}") 