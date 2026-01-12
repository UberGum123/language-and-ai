from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from reader import Reader
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)
import pandas as pd
from logger import Logger

class Modeler:
    """ 
    Chooses a model between SVM and BERT based on user input.
    Runs the chosen model on the preprocessed dataset and outputs results.

        Args:
            dataset: Preprocessed dataset with folds.
            (Optional) hyperparameters: a list of C's for the SVM model.
    """    
    def train_svm(dataset, hyperparameters):
        logger = Logger(log_file="svm_training_results.log")
        
        label_encoder = LabelEncoder()
        label_encoder.fit(["left", "right", "center"])
        C = hyperparameters
        
        class_names = label_encoder.classes_
        for c_value in C:
            logger.reset_stats()
            logger.log_configuration(f"SVM", {"C": c_value})
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
                    max_iter=100000
                )

                clf.fit(X_train_vec, y_train)

                y_pred = clf.predict(X_val_vec)

                logger.log_per_fold(fold['fold_id'], y_val, y_pred, class_names)
                
            logger.log_overall_performance(class_names)

