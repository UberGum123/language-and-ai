from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from src.reader import Reader
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)
import pandas as pd
from src.logger import Logger
from src.visualizations import Visualizer

class Modeler:
    """ 
    Chooses a model between SVM and BERT based on user input.
    Runs the chosen model on the preprocessed dataset and outputs results.

        Args:
            dataset: Preprocessed dataset with folds.
            (Optional) hyperparameters: a list of C's for the SVM model.
    """    
    @staticmethod
    def train_svm(dataset, hyperparameters):
        logger = Logger(log_file="output/svm_training_results.log")
        visualizer = Visualizer()
        
        label_encoder = LabelEncoder()
        label_encoder.fit(["left", "right", "center"])
        C = hyperparameters
        
        class_names = label_encoder.classes_
        
        # Track best configuration
        best_config = {
            'C': None,
            'mean_macro_f1': 0,
            'model': None,
            'vectorizer': None
        }
        
        for c_value in C:
            logger.reset_stats()
            logger.log_configuration(f"SVM", {"C": c_value})
            
            #Auxillary stuff for visualizations
            last_model = None
            last_vectorizer = None
            last_X_val = None
            last_y_val = None
            
            for fold in dataset.folds:
                print(f"\nFold {fold['fold_id']}")

                X_train = fold["train"]      
                X_val = fold["val"]

                y_train = label_encoder.transform(fold["train_labels"])
                y_val = label_encoder.transform(fold["val_labels"])

                # Vectorizer that accepts tokens directly
                vectorizer = TfidfVectorizer(
                    analyzer=lambda x: x,   
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                    norm="l2"
                )

                X_train_vec = vectorizer.fit_transform(X_train)
                X_val_vec = vectorizer.transform(X_val)

                clf = LinearSVC(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=10000
                )

                clf.fit(X_train_vec, y_train)

                y_pred = clf.predict(X_val_vec)

                logger.log_per_fold(fold['fold_id'], y_val, y_pred, class_names)
                
                last_model = clf
                last_vectorizer = vectorizer
                last_X_val = X_val_vec
                last_y_val = y_val
                
            logger.log_overall_performance(class_names)
            
            mean_macro_f1 = np.mean(logger.macro_f1_scores)
            
            if mean_macro_f1 > best_config['mean_macro_f1']:
                best_config['C'] = c_value
                best_config['mean_macro_f1'] = mean_macro_f1
                best_config['model'] = last_model
                best_config['vectorizer'] = last_vectorizer
                
            # Generate visualizations for this C value
            if visualizer:
                print(f"\n{'='*60}")
                print(f"GENERATING VISUALIZATIONS FOR C={c_value}")
                print(f"{'='*60}\n")
                
                # Mean Confusion Matrix
                mean_cm = np.mean(logger.confusion_matrices, axis=0)
                visualizer.plot_confusion_matrix(
                    mean_cm, 
                    class_names, 
                    title=f'Mean Confusion Matrix (C={c_value})'
                )
                
                # Metrics Summary
                metrics_dict = {
                    'Macro F1': logger.macro_f1_scores,
                    'Weighted F1': logger.weighted_f1_scores,
                    'MCC': logger.mcc_scores,
                    'Accuracy': logger.fold_accuracies
                }
                visualizer.plot_metrics_summary(
                    metrics_dict, 
                    title=f"Metrics Across Folds (C={c_value})"
                )
                
                # Top Features
                visualizer.plot_top_features(
                    last_vectorizer,
                    last_model,
                    class_names,
                    top_n=15,
                    title = f"Top Features per Class (C={c_value})"
                )
                
                # PCA Embeddings
                visualizer.plot_pca_embeddings(
                    last_X_val,
                    last_y_val,
                    class_names,
                    title=f"PCA of Validation Set (C={c_value})"
                )
                
        return best_config

