import joblib
from pathlib import Path
from reader import Reader

class DatasetSaver:
    def save_dataset(dataset, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dataset, path)

    def load_dataset(path):
        path = Path(path)
        return joblib.load(path)

reader = Reader(
    min_freq=5,
    remove_stopwords=True,
    use_lemmatization=True,
    use_bigrams=False,
    n_splits=5
)

dataset = reader.load_and_preprocess_csv(
    "data/political_leaning.csv",
    text_column="post",
    label_column="political_leaning"
)

DatasetSaver.save_dataset(dataset, "cache/reddit_dataset.joblib")