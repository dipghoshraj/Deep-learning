import numpy as np
import torch

class CustomeDataLoader:
    def __init__(self, memmap_path: str, block_size: int, dtype = np.int32, start=0, end=None):
        self.tokens =  np.memmap(memmap_path, dtype=dtype, mode='r')
        self.block_size = block_size
        self.total_tokens = len(self.tokens)

        self.end = len(self.tokens) - block_size if end is None else end
        self.start = start
        self.length = self.end - self.start


    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        idx += self.start
        if idx + self.block_size + 1 > self.end:
            raise IndexError("Index out of bounds.")
        block = self.tokens[idx : idx + self.block_size + 1]
        x = torch.tensor(block[:-1], dtype=torch.long)
        y = torch.tensor(block[1:], dtype=torch.long)
        return x, y
    

    def collate_fn(batch):
        x = torch.stack([item[0] for item in batch])
        y = torch.stack([item[1] for item in batch])
        return x, y



class DataLoader:
    def __init__(self, tokens, batch_size, context_length) -> None:
        self.tokens = tokens
        self.batch_size = batch_size
        self.context_length = context_length

        self.current_position = 0

    def get_batch(self) -> torch.tensor:
        b, c = self.batch_size, self.context_length

        start_pos = self.current_position
        end_pos = self.current_position + b * c + 1

        # if the batch exceeds total length, get the data till last token
        # and take remaining from starting token to avoid always excluding some data
        add_data = -1 # n, if length exceeds and we need `n` additional tokens from start
        if end_pos > len(self.tokens):
            add_data = end_pos - len(self.tokens)
            end_pos = len(self.tokens)

        d = self.tokens[start_pos:end_pos]
        if add_data != -1:
            d = torch.cat([d, self.tokens[:add_data]])

        x = (d[:-1]).view(b, c)  # inputs
        y = (d[1:]).view(b, c)  # targets

        self.current_position += b * c # set the next position
        if self.current_position > len(self.tokens) - 1:
            self.current_position = 0
        return x, y