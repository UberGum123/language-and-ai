import re
import unicodedata
import pandas as pd
from pandarallel import pandarallel
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist
from tqdm import tqdm


# Download NLTK resources (once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


class Dataset:
    """
    Used for storing the data.
    """
    def __init__(self, df):
        self.df = df.copy()
        self.processed = None


class Reader:
    """
    Handles loading CSV data, preprocessing posts, and splitting datasets.
    """

    def __init__(self,
                 min_freq=5,
                 remove_stopwords=True,
                 use_lemmatization=True,
                 use_bigrams=False,
                 random_state=42):

        self.min_freq = min_freq
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.use_bigrams = use_bigrams
        self.random_state = random_state
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    # ---------------------------
    # Preprocessing methods
    # ---------------------------

    def _remove_formatting(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def _expand_contractions(self, text):
        if not isinstance(text, str):
            return ''
        
        if "'" not in text:  # skip if no apostrophes
            return text
        
        # Prevent errors from occurring when reading a foreign character
        text_ascii = text.encode('ascii', errors='ignore').decode()
        
        try:
            return contractions.fix(text_ascii)
        except Exception:
            return text  # if removing the contractions still fails, just return the normal text
    
    def _remove_non_alpha(self, tokens):
        return [t for t in tokens if t.isalpha()]
    
    def _normalize_tokens(self, tokens):
        normalized = []
        for t in tokens:
            t = t.lower()
            t = ''.join(
                c for c in unicodedata.normalize('NFKD', t)
                if not unicodedata.combining(c)
            )
            normalized.append(t)
        return normalized

    def _remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stop_words]

    def _get_wordnet_pos(self, tag):
        if tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def _lemmatize(self, tokens):
        pos_tags = pos_tag(tokens)
        return [
            self.lemmatizer.lemmatize(word, self._get_wordnet_pos(tag))
            for word, tag in pos_tags
        ]

    def _frequency_cutoff(self, corpus):
        """
        Remove tokens with frequency < min_freq
        """
        all_tokens = [t for doc in corpus for t in doc]
        freq = FreqDist(all_tokens)
        filtered = [
            [t for t in doc if freq[t] >= self.min_freq]
            for doc in corpus
        ]
        return filtered

    def _extract_bigrams(self, corpus):
        all_tokens = [t for doc in corpus for t in doc]
        finder = BigramCollocationFinder.from_words(all_tokens)
        finder.apply_freq_filter(self.min_freq)
        bigrams = finder.nbest(BigramAssocMeasures.pmi, 20)
        return bigrams

    def preprocess_post(self, text):
        """
        Preprocesses the text of a single post.
        """
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

    def load_and_preprocess_csv(self, csv_path, text_column='post'):
        """
        Loads CSV, preprocesses posts and returns Dataset object.
        """
        df = pd.read_csv(csv_path)
        dataset = Dataset(df)

        # Wrap tqdm around pandas apply
        tqdm.pandas(desc="Preprocessing posts")
        dataset.df['processed'] = dataset.df[text_column].progress_apply(self.preprocess_post)
        dataset.processed = dataset.df['processed'].tolist()

        # Apply frequency cutoff
        dataset.processed = self._frequency_cutoff(dataset.processed)

        # Optionally extract corpus-wide bigrams
        if self.use_bigrams:
            print("Extracting corpus-wide bigrams...")
            dataset.bigrams = self._extract_bigrams(dataset.processed)
        else:
            dataset.bigrams = None

        return dataset
