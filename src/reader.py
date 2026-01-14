import re
import unicodedata
import pandas as pd
import nltk
import contractions
from tqdm import tqdm

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist
from sklearn.model_selection import StratifiedKFold

from utils.data_saver import DatasetSaver

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


class Dataset:
    """
    Container for dataset states and folds.
    """
    def __init__(self, df):
        self.df = df
        self.processed = None
        self.bigrams = None
        self.folds = []  # list of dicts, one per fold


class Reader:
    """
    Loads CSV data, preprocesses text, and creates stratified k-fold splits.
    """

    def __init__(
        self,
        n_splits,
        min_freq=5,
        remove_stopwords=True,
        use_lemmatization=True,
        use_bigrams=False,
        random_state=42
    ):
        self.min_freq = min_freq
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.use_bigrams = use_bigrams
        self.n_splits = n_splits
        self.random_state = random_state

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Precompile regex
        self.whitespace_re = re.compile(r'\s+')

    # ---------------------------
    # Preprocessing helpers
    # ---------------------------

    def _remove_formatting(self, text):
        return self.whitespace_re.sub(' ', text).strip()

    def _expand_contractions(self, text):
        if "'" not in text:
            return text

        text = text.encode("ascii", errors="ignore").decode()

        try:
            return contractions.fix(text)
        except Exception:
            return text

    def _normalize_tokens(self, tokens):
        return [
            ''.join(
                c for c in unicodedata.normalize('NFKD', t.lower())
                if not unicodedata.combining(c)
            )
            for t in tokens
        ]

    def _remove_non_alpha(self, tokens):
        return [t for t in tokens if t.isalpha()]

    def _remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stop_words]

    def _lemmatize(self, tokens):
        # Noun-only lemmatization
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    # ---------------------------
    # Core preprocessing
    # ---------------------------

    def preprocess_post(self, text):
        text = self._remove_formatting(text)
        text = self._expand_contractions(text)

        tokens = word_tokenize(text)
        tokens = self._normalize_tokens(tokens)
        tokens = self._remove_non_alpha(tokens)

        if self.remove_stopwords:
            tokens = self._remove_stopwords(tokens)

        if self.use_lemmatization:
            tokens = self._lemmatize(tokens)

        return tokens

    # ---------------------------
    # Corpus-level operations
    # ---------------------------

    def _frequency_cutoff(self, corpus):
        all_tokens = [t for doc in corpus for t in doc]
        freq = FreqDist(all_tokens)

        return [
            [t for t in doc if freq[t] >= self.min_freq]
            for doc in corpus
        ]

    def _extract_bigrams(self, corpus):
        all_tokens = [t for doc in corpus for t in doc]
        finder = BigramCollocationFinder.from_words(all_tokens)
        finder.apply_freq_filter(self.min_freq)
        return finder.nbest(BigramAssocMeasures.pmi, 20)

    # ---------------------------
    # Public API
    # ---------------------------

    def load_and_preprocess_csv(
        self,
        csv_path,
        text_column,
        label_column,
        load_from_existing
    ):
        if load_from_existing:
            return DatasetSaver.load_dataset("cache/political_leaning.joblib")
        df = pd.read_csv(csv_path)
        df = df.truncate(after=1000)  # For testing purposes only (speed up)
        dataset = Dataset(df)
        # --------
        # Preprocessing
        # --------
        processed = []
        for text in tqdm(df[text_column], desc="Preprocessing posts"):
            processed.append(self.preprocess_post(text))

        dataset.df['processed'] = processed
        dataset.processed = processed

        # Frequency cutoff
        dataset.processed = self._frequency_cutoff(dataset.processed)

        # Optional bigrams
        if self.use_bigrams:
            dataset.bigrams = self._extract_bigrams(dataset.processed)

        # --------
        # Stratified K-Fold CV
        # --------
        X = dataset.processed
        y = df[label_column].tolist()

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        for fold_id, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            fold = {
                "fold_id": fold_id,
                "train": [X[i] for i in train_idx],
                "val": [X[i] for i in val_idx],
                "train_labels": [y[i] for i in train_idx],
                "val_labels": [y[i] for i in val_idx]
            }
            dataset.folds.append(fold)

        return dataset
    
    def load_and_preprocess_for_bert(self, csv_path, text_column, label_column, text_pair_column=None):
        """
        Preprocessing for BERT (Uncased) based on Section 3 of Devlin et al. (2018).
        
        Args:
            text_pair_column (str): Optional. The column name for the second sentence 
                                    if the task involves sentence pairs (e.g., Q&A, Entailment).
        """
        df = pd.read_csv(csv_path)
        
        # Helper to normalize text for uncased BERT
        def clean_text_for_uncased(text):
            if pd.isna(text): return ""
            text = str(text).lower()
            # Remove accents (NFD normalization splits characters from accents)
            text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
            return text

        # Apply normalization
        df[text_column] = df[text_column].apply(clean_text_for_uncased)
        dataset = Dataset(df)

        if text_pair_column:
            df[text_pair_column] = df[text_pair_column].apply(clean_text_for_uncased)
            # Store tuple of (text_a, text_b)
            dataset.processed = list(zip(df[text_column].tolist(), df[text_pair_column].tolist()))
        else:
            # Store simple list of strings
            dataset.processed = df[text_column].tolist()
                
        # Create stratified folds
        X = dataset.processed
        y = df[label_column].tolist()
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for fold_id, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            fold = {
                "fold_id": fold_id,
                "train": [X[i] for i in train_idx],  
                "val": [X[i] for i in val_idx],      
                "train_labels": [y[i] for i in train_idx],
                "val_labels": [y[i] for i in val_idx]
            }
            dataset.folds.append(fold)
        
        return dataset
