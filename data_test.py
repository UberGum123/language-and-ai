# Example usage of the reader.py file
from src.reader import Reader

# Initialize the Reader
reader = Reader(n_splits=5)

# Load CSV and preprocess
dataset = reader.load_and_preprocess_csv('data/political_leaning.csv')

print(dataset.df.head())

for fold in dataset.folds:
    print(
        fold["fold_id"],
        len(fold["train"]),
        len(fold["val"])
    )