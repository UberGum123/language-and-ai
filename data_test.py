# Example usage of the reader.py file
from reader import Reader

# Initialize the Reader
reader = Reader(min_freq=2, use_lemmatization=True, use_bigrams=True)

# Load CSV and preprocess
dataset = reader.load_and_preprocess_csv('data/political_leaning.csv')

print(dataset.df.head())