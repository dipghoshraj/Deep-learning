from datasets import load_from_disk


class DataLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = None

    def load_dataset(self):
        self.dataset = load_from_disk(self.dataset_path)
        print(f"Dataset loaded from {self.dataset_path}")

    def get_dataset(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        return self.dataset
    
    def flatten(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        # self.dataset
    

