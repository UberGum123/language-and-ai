import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    accuracy_score
)

class Logger:
    """
    A simple logger class to log messages to console and file. 
    Called by the Modeller class to log the experiments' progress and statistics.
    """
    def __init__(self, log_file='output/experiment.log'):
        self.log_file = log_file
                
        #Clear/Create the log file at the start
        with open(self.log_file, 'w') as f:
            f.write("Experiment Log\n")
            
        self.reset_stats()
    
    def reset_stats(self):
        """
        Resets the lists. Call this before starting a new hyperparameter loop.
        """
        self.fold_accuracies = []
        self.macro_f1_scores = []
        self.weighted_f1_scores = []
        self.mcc_scores = []
        self.confusion_matrices = []

    def write_logs_to_file(self, message):
        """
        Writes the logged metrics/message to the specified log file.
        """
        try:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
        except IOError as e:
            print(f"Error writing to log file: {e}")

    def _print_and_save(self, message):
        """
        Helper to print to console and save to file simultaneously.
        """
        print(message)
        self.write_logs_to_file(message)

    def log_per_fold(self, fold_id, y_true, y_pred, class_names):
        """
        Calculates, logs, and stores performance metrics for a specific fold.
        """
        #Calculate Metrics 
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        mcc = matthews_corrcoef(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        #Store Metrics 
        self.macro_f1_scores.append(macro_f1)
        self.weighted_f1_scores.append(weighted_f1)
        self.mcc_scores.append(mcc)
        self.fold_accuracies.append(acc)
        self.confusion_matrices.append(cm)

        #Generate Output String
        output = []
        output.append(f"\nFold {fold_id}")
        output.append(f"Macro F1:{macro_f1:.4f}")
        output.append(f"Weighted F1:{weighted_f1:.4f}")
        output.append(f"MCC:{mcc:.4f}")
        output.append(f"Accuracy:{acc:.4f}")
        
        #Confusion Matrix formatting
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        output.append("\n Confusion Matrix (rows=true, cols=pred):")
        output.append(str(cm_df))

        #Classification Report
        report = classification_report(y_true, y_pred, target_names=class_names)
        output.append("\nClassification Report:")
        output.append(report)

        #Print and Save
        full_message = "\n".join(output)
        self._print_and_save(full_message)
        
    def log_configuration(self, config_name, params):
        """
        Logs a header for the specific hyperparameter setting being run.
        """
        msg = f"\nRunning Configuration: {config_name}\nParameters: {params}\n{'='*40}"
        self._print_and_save(msg)
        
    def log_overall_performance(self, class_names):
        """
        Calculates means/stds of stored metrics and logs the overall summary.
        """
        if not self.fold_accuracies:
            print("No folds have been logged yet.")
            return

        output = []
        output.append(f"\nCross-Validation Summary")        
        # Accuracy Stats
        output.append(f"Mean Accuracy: {np.mean(self.fold_accuracies):.4f}")
        output.append(f"Std Accuracy:  {np.std(self.fold_accuracies):.4f}")
        
        #F1 and MCC Stats
        output.append("\nCross-Validation Summary for F1 and MCC")
        output.append(f"Macro F1: {np.mean(self.macro_f1_scores):.4f} +- {np.std(self.macro_f1_scores):.4f}")
        output.append(f"Weighted F1:{np.mean(self.weighted_f1_scores):.4f} +- {np.std(self.weighted_f1_scores):.4f}")
        output.append(f"MCC:{np.mean(self.mcc_scores):.4f} +- {np.std(self.mcc_scores):.4f}")

        #Mean Confusion Matrix
        mean_cm = np.mean(self.confusion_matrices, axis=0)
        mean_cm_df = pd.DataFrame(mean_cm, index=class_names, columns=class_names)
        
        output.append("\nMean Confusion Matrix Across Folds")
        output.append("(values are averages)")
        output.append(str(mean_cm_df))

        #Print and Save
        full_message = "\n".join(output)
        self._print_and_save(full_message)