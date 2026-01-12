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
    