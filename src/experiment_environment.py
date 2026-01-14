from src.descriptive_statistics import Descriptives
from src.reader import Reader
import pandas as pd
import re
import geonamescache


from src.model import Modeler
from utils.data_saver import DatasetSaver

# Parses user input on the experiment setup.
# Accepts user input in a JSON schema 

class ExperimentEnvironment:
    """ Parses user configuration, sets up the experiment environment accordingly, and carries out the experiment. """
    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.descriptive_stats = Descriptives()
    
    def config_sanity_checks(self):
        mandatory_fields = ['data', 'descriptive_statistics', 'modeling', 'preprocessing', 'masking']
        for field in mandatory_fields:
            if field not in self.config:
                raise ValueError(f"Missing mandatory configuration field: {field}")
        
        mandatory_data_fields = ['file_path', 'text_column', 'label_column', 'author_column']
        for field in mandatory_data_fields:
            if field not in self.config['data']:
                raise ValueError(f"Missing mandatory data configuration field: {field}")
        
        mandatory_descriptive_fields = ['enabled', 'plot_label_distribution', 'word_count_per_label', 'get_author_info', 'get_top_n_words_per_label']
        for field in mandatory_descriptive_fields:
            if field not in self.config['descriptive_statistics']:
                raise ValueError(f"Missing mandatory descriptive statistics configuration field: {field}")
        
        mandatory_modeling_fields = ['enabled', 'model_type', 'hyperparameters']
        for field in mandatory_modeling_fields:
            if field not in self.config['modeling']:
                raise ValueError(f"Missing mandatory modeling configuration field: {field}")
    
        mandatory_preprocessing_fields = ['number_of_folds', 'get_descriptive_statistics_after_preprocessing']
        for field in mandatory_preprocessing_fields:
            if field not in self.config['preprocessing']:
                raise ValueError(f"Missing mandatory preprocessing configuration field: {field}")
        
    def preprocess_data(self):
        print(f"\n Starting data preprocessing")
        get_descriptive_stats_after_preprocessing = self.config['preprocessing']['get_descriptive_statistics_after_preprocessing']
        text_column = self.config['data']['text_column']
        label_column = self.config['data']['label_column']
        author_column = self.config['data']['author_column']
        filepath = self.config['data']['file_path']
        
        number_of_folds = self.config['preprocessing']['number_of_folds']
        load_from_existing = self.config['load_dataset']
        reader = Reader(n_splits=number_of_folds)
        processed_dataset = reader.load_and_preprocess_csv(
            filepath,
            text_column=text_column,
            label_column=label_column,
            author_column=author_column,
            load_from_existing=load_from_existing
        )
        
        print(f"\n Data preprocessing complete.")
        if get_descriptive_stats_after_preprocessing:
            print("Outputting descriptive statistics for the processed data")
            raise NotImplementedError("Descriptive statistics after preprocessing not yet implemented.")
            #TODO: Implement descriptive statistics for processed data
        return processed_dataset
            
    
    def get_descriptive_stats(self):

        descriptive_config = self.config['descriptive_statistics']
        enabled = descriptive_config['enabled']
        if not enabled:
            return
        
        # Get everythin that we need from the JSON config
        plot_label_distribution = descriptive_config['plot_label_distribution']
        word_count_per_label = descriptive_config['word_count_per_label']
        get_author_info = descriptive_config['get_author_info']
        get_top_n_words_per_label = descriptive_config['get_top_n_words_per_label']
        label_column = self.config['data']['label_column']
        text_column = self.config['data']['text_column']
        author_column = self.config['data']['author_column']
        
        if plot_label_distribution:
            self.descriptive_stats.plot_label_distribution(
                self.dataset,
                label_column
            )
        if word_count_per_label:
            self.descriptive_stats.word_count_per_label(
                self.dataset,
                text_column,
                label_column
            )
        if get_author_info:
            self.descriptive_stats.get_author_info(
                self.dataset,
                author_column
            )
        if get_top_n_words_per_label is not None:
            self.descriptive_stats.get_top_words_per_leaning(
                self.dataset,
                get_top_n_words_per_label
            )

    def train_model(self, dataset):
        modelling_config = self.config['modeling']
        enabled = modelling_config['enabled']
        enable_vis = modelling_config.get('enable_visualizations')
        if not enabled:
            return
        model_type = modelling_config['model_type']
        hyperparameters = modelling_config['hyperparameters']
        if model_type == "SVM":
            return Modeler.train_svm(dataset, hyperparameters['C'])
        elif model_type == 'BERT':
            hyperparams = modelling_config.get('hyperparameters', {})
            print(f"Training BERT with hyperparameters: {hyperparams}")
            
            self.best_model_info = Modeler.train_bert(
                dataset,
                hyperparams,
                enable_visualizations=enable_vis
            )
        else:
            raise ValueError(f"Wrong model type: {model_type}, choose between SVM and BERT")
        
    def mask(self, dataset: pd.DataFrame) -> pd.DataFrame:
        print('Masking data')
        masking_config = self.config['masking']

        if not masking_config.get('enabled', False):
            print("Masking disabled, returning dataset unchanged.")
            return dataset

        strategy = masking_config.get('masking_strategy', 0)
        text_column = self.config['data']['text_column']

        print(f"Applying masking strategy: {strategy}")

        # Ideology/party keywords
        regex_keywords_pattern = r"\b(?:" \
            r"cons|conser|conserv|conservative|" \
            r"lib|liber|liberal|liberter|" \
            r"prog|progressive|" \
            r"leftist|lefty|left[-\s]?wing|" \
            r"righty|right[-\s]?wing|" \
            r"far[-\s]?(left|right)|alt[-\s]?(right|left)|" \
            r"libertarian|centrist|moderate|socialist|anarchist|communist|marxist|" \
            r"dem|demo|democrat|democrats|democratic|" \
            r"repub|republican|republicans|" \
            r"gop|dnc" \
            r")\b"

        regex_keywords = re.compile(regex_keywords_pattern)

        # Political figures (with optional first names)
        regex_figures_pattern = r"\b(?:" \
            r"(?:donald )?trump|" \
            r"(?:joe )?biden|" \
            r"(?:barack )?obama|" \
            r"(?:george w\. )?bush|" \
            r"(?:bill )?clinton|" \
            r"(?:hillary )?clinton|" \
            r"(?:kamala )?harris|" \
            r"(?:mike )?pence|" \
            r"(?:bernie )?sanders|" \
            r"(?:alexandria ocasio-)?cortez|aoc|" \
            r"(?:nancy )?pelosi|" \
            r"(?:mitch )?mcconnell|" \
            r"(?:boris )?johnson|" \
            r"(?:theresa )?may|" \
            r"(?:rishi )?sunak|" \
            r"(?:keir )?starmer|" \
            r"(?:vladimir )?putin|" \
            r"(?:xi )?jinping|" \
            r"(?:narendra )?modi|" \
            r"(?:emmanuel )?macron|" \
            r"(?:angela )?merkel|" \
            r"(?:justin )?trudeau|" \
            r"(?:jair )?bolsonaro|" \
            r"(?:recep tayyip )?erdogan|" \
            r"(?:scott )?morrison|" \
            r"(?:jacinda )?ardern" \
            r")\b"

        regex_figures = re.compile(regex_figures_pattern)

        # Other political words / phrases
        regex_political_words_pattern = r"\b(?:" \
            r"abortion|guns|immigration|taxes|healthcare|" \
            r"medicare|medicaid|climate|environment|protest|" \
            r"democracy|socialism|capitalism|freedom|rights|" \
            r"marriage|equality|vote|voting|election|" \
            r"police|crime|war|military|maga|woke|" \
            r"libtard|snowflake|commie|social justice warrior|sjw|" \
            r"redneck|redpill|gay|trans|queer|immigrant" \
            r"beta male|soyboy|cuck|karen|" \
            r"FBI|FSB|KGB|ICE|CIA|IDF" \
            r")\b"

        regex_political_words = re.compile(regex_political_words_pattern)

        gc = geonamescache.GeonamesCache()
        countries = set([c['name'].lower() for c in gc.get_countries().values()])
        cities = set([c['name'].lower() for c in gc.get_cities().values()])

        def mask_string(text: str) -> str:
            if not isinstance(text, str):
                return text

            # Strategy 1+: remove ideology keywords
            if strategy >= 1:
                text = regex_keywords.sub("", text)

            # Strategy 2+: remove political figures
            if strategy >= 2:
                text = regex_figures.sub("", text)

            # Strategy 3+: remove political words / phrases
            if strategy >= 3:
                text = regex_political_words.sub("", text)

            # Strategy 4+: remove locations
            if strategy >= 4:
                tokens = text.split()
                tokens = [w for w in tokens if w.lower() not in countries and w.lower() not in cities]
                text = " ".join(tokens)

            # Remove extra spaces left by deletions
            text = re.sub(r"\s+", " ", text).strip()

            return text

        dataset.df[text_column] = (
        dataset.df[text_column].astype(str).apply(mask_string)
        )

        if masking_config.get('get_descriptive_statistics_after_masking', False):
            print("Descriptive statistics after masking not implemented yet.")

        print("Masking complete")

        return dataset
    
    def run(self):
        try:
            # Validate configuration
            self.config_sanity_checks()
            print("Configuration is valid.")
            # Load raw dataset
            data_config = self.config['data']
            filepath = data_config['file_path']
            self.dataset = pd.read_csv(filepath)
            print("Raw dataset is loaded")

            # Run descriptive statistics on raw data
            self.get_descriptive_stats()
            
            # Preprocess data (if modeling is enabled)
            processed_dataset = self.preprocess_data()
            # DatasetSaver.save_dataset(processed_dataset, "cache/political_leaning.joblib")
            # Run modeling
            # if processed_dataset is not None:
            #     model = self.train_model(processed_dataset)
            
            # Run masking based on the integer code defined in the config json file
            masked_dataset = self.mask(processed_dataset)
            # DatasetSaver.save_dataset(masked_dataset, "cache/preprocessed_masked.joblib")

            if masked_dataset is not None:
                masked_model = self.train_model(masked_dataset)

            print("EXPERIMENT COMPLETE")
            
        except Exception as e:
            print(f"\n Exception during the experiment: {e}")
            raise