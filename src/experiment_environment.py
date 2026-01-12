from descriptive_statistics import Descriptives
from reader import Reader
import pandas as pd

# Parses user input on the experiment setup.
# Accepts user input in a JSON schema format    
# {
#     "data": {
#         "file_path": "data/political_leaning.csv",
#         "text_column": "post",
#         "label_column": "political_leaning",
#         "author_column": "author_id"
#      },
#     "descriptive_statistics": {
#         "enabled": true,
#         "plot_label_distribution": true,
#         "word_count_per_label": true,
#         "get_author_info": true,
#         "get_top_n_words_per_label": n
#      } ,
#     "modeling": {
#         "enabled": false
#         "model_type": "SVM", "BERT"
#         "hyperparameters": {
#             "C": [0.1, 1, 10],
#      },
#     "preprocessing": {
#         "number_of_folds": 5,
#         "get_descriptive_statistics_after_preprocessing": true
#      },
#     "masking": {
#         "enabled": false,
#         "get_descriptive_statistics_after_masking": true,
#         "masking_strategy": "entity_masking", "random_masking"
#      }
#}

class ExperimentEnvironment:
    """ Parses user configuration, sets up the experiment environment accordingly, and carries out the experiment. """
    def __init__(self, config):
        self.config = config
        self.dataset = None
    
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
        
    
    def load_data(self):
        #TODO: Implement data loading based on config here,if not enabled: return. jsut load, no preprocessing yet
        pass
    
    def preprocess_data(self):
        #TODO: Implement data preprocessing based on config here
        pass
    
    def get_descriptive_stats(self):
        #TODO: Implement descriptive statistics based on config here
        pass
    
    def train_model(self):
        #TODO: Implement model training based on config here
        pass
    
    def mask(self):
        #TODO: Implement masking based on config here, and also the masking functionality in the reader
        pass
    
    def run(self):
        try:
            # Validate configuration
            self.validate_config()
            print("Configuration is valid.")
            
            # Load data
            self.load_data()
            
            # Run descriptive statistics on raw data
            self.run_descriptive_statistics()
            
            # Preprocess data (if modeling is enabled)
            processed_dataset = self.run_preprocessing()
            
            # Run modeling
            if processed_dataset is not None:
                self.run_modeling(processed_dataset)
            
            # Run masking experiments
            self.run_masking()

            print("EXPERIMENT COMPLETE")
            
        except Exception as e:
            print(f"\n Exception during the experiment: {e}")
            raise