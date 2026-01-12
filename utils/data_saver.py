import joblib
from pathlib import Path

class DatasetSaver:
    """Helper class to save and load datasets to save up time for the experiment trials (preprocessing takes 20 mins)"""
    def save_dataset(dataset, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dataset, path)
        print(f"Dataset saved to {path}")

    def load_dataset(path):
        path = Path(path)
        print(f"Loading dataset from {path}")
        return joblib.load(path)
