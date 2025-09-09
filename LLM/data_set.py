from data.data_builder import DataLoader

data_loader = DataLoader("../tokenized_sql_dataset")
data_loader.load_dataset()

dataset = data_loader.get_dataset()
print(dataset)