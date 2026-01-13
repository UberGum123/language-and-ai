from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from transformers import Trainer
import torch
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
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset as TorchDataset

class TokenListDataset(TorchDataset):
    """
    PyTorch Dataset that converts token lists to BERT input.
    """
    def __init__(self, token_lists, labels, tokenizer, max_length=128):

        self.token_lists = token_lists
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.token_lists)
    
    def __getitem__(self, idx):
        # Convert token list back to string (space-separated)
        text = ' '.join(self.token_lists[idx])
        
        # Tokenize with BERT tokenizer
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Compute metrics for evaluation during training."""
    from sklearn.metrics import accuracy_score, f1_score
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_weighted': f1_score(labels, predictions, average='weighted')
    }


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

    @staticmethod
    def train_bert(dataset, hyperparameters, enable_visualizations=True):

        print("Starting BERT training...")
        
        logger = Logger(log_file="bert_training_results.log")
        visualizer = Visualizer() if enable_visualizations else None
        
        # Extract hyperparameters
        model_name = hyperparameters.get('model_name', 'bert-base-uncased')
        learning_rate = hyperparameters.get('learning_rate', 2e-5)
        epochs = hyperparameters.get('epochs', 3)
        batch_size = hyperparameters.get('batch_size', 16)
        max_length = hyperparameters.get('max_length', 128)
        
        print(f"Model: {model_name}")
        print(f"Learning rate: {learning_rate}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Max length: {max_length}\n")
        
        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Label encoding
        label_encoder = LabelEncoder()
        label_encoder.fit(["left", "right", "center"])
        class_names = label_encoder.classes_
        
        logger.log_configuration("BERT", hyperparameters)
        
        # Train on each fold
        for fold in dataset.folds:
            print(f"Fold {fold['fold_id']}")
  
            # Get token lists
            X_train = fold["train"]  # List of token lists
            X_val = fold["val"]
            y_train = label_encoder.transform(fold["train_labels"])
            y_val = label_encoder.transform(fold["val_labels"])
            
            print(f"Train samples: {len(X_train)}")
            print(f"Val samples: {len(X_val)}")
            
            # Create datasets
            train_dataset = TokenListDataset(X_train, y_train, tokenizer, max_length)
            val_dataset = TokenListDataset(X_val, y_val, tokenizer, max_length)
            
            # Initialize model
            model = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(class_names)
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f'./results/bert_fold_{fold["fold_id"]}',
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",
                logging_dir=f'./logs/bert_fold_{fold["fold_id"]}',
                logging_steps=100,
                save_total_limit=2,
                report_to="none" 
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            
            # Train
            print(f"\nTraining BERT on fold {fold['fold_id']}...")
            trainer.train()
            
            # Evaluate
            print(f"\nEvaluating on validation set...")
            predictions = trainer.predict(val_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            
            # Log metrics
            logger.log_per_fold(fold['fold_id'], y_val, y_pred, class_names)
        
        # Overall performance
        logger.log_overall_performance(class_names)
        
        # Visualizations
        if visualizer:
            print(f"\n{'='*60}")
            print(f"GENERATING VISUALIZATIONS")
            print(f"{'='*60}\n")
            
            mean_cm = np.mean(logger.confusion_matrices, axis=0)
            visualizer.plot_confusion_matrix(
                mean_cm, 
                class_names, 
                title='Mean Confusion Matrix (BERT)'
            )
            
            metrics_dict = {
                'Macro F1': logger.macro_f1_scores,
                'Weighted F1': logger.weighted_f1_scores,
                'MCC': logger.mcc_scores,
                'Accuracy': logger.fold_accuracies
            }
            visualizer.plot_metrics_summary(
                metrics_dict, 
                title="BERT Metrics Across Folds"
            )
                    
        # Return best config
        mean_macro_f1 = np.mean(logger.macro_f1_scores)
        
        print("Finished training BERT: ")
        print(f"Mean Macro F1: {mean_macro_f1:.4f}")
        print(f"Mean Accuracy: {np.mean(logger.fold_accuracies):.4f}")
        
        return {
            'model_name': model_name,
            'mean_macro_f1': mean_macro_f1,
            'mean_accuracy': np.mean(logger.fold_accuracies)
        }