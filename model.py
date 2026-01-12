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
from data_saver import DatasetSaver


# Load CSV and preprocess
dataset = DatasetSaver.load_dataset("cache/reddit_dataset.joblib")
print(len(dataset.folds))            # Should be >0
print(type(dataset.folds[0]))        # Should be dict
print(dataset.folds[0].keys()) 

def train_svm_from_tokens(dataset):

    label_encoder = LabelEncoder()
    label_encoder.fit(["left", "right", "center"])

    fold_accuracies = []

    macro_f1_scores = []
    weighted_f1_scores = []
    mcc_scores = []
    confusion_matrices = []
    
    class_names = label_encoder.classes_

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
            C=1.0,
            class_weight="balanced",
            max_iter=100000
        )

        clf.fit(X_train_vec, y_train)

        y_pred = clf.predict(X_val_vec)

        macro_f1 = f1_score(y_val, y_pred, average="macro")
        weighted_f1 = f1_score(y_val, y_pred, average="weighted")
        mcc = matthews_corrcoef(y_val, y_pred)

        macro_f1_scores.append(macro_f1)
        weighted_f1_scores.append(weighted_f1)
        mcc_scores.append(mcc)

        print(f"Macro F1:    {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(f"MCC:         {mcc:.4f}")
        
        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)

        print(f"Accuracy: {acc:.4f}")
        cm = confusion_matrix(y_val, y_pred)
        confusion_matrices.append(cm)
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(pd.DataFrame(cm, index=class_names, columns=class_names))

        print(
            classification_report(
                y_val,
                y_pred,
                target_names=class_names
            )
        )
    print("\n Cross-Validation Summary")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Std Accuracy:  {np.std(fold_accuracies):.4f}")
    print("\nCross-Validation Summary for F1 and MCC")
    print(f"Macro F1:    {np.mean(macro_f1_scores):.4f} ± {np.std(macro_f1_scores):.4f}")
    print(f"Weighted F1: {np.mean(weighted_f1_scores):.4f} ± {np.std(weighted_f1_scores):.4f}")
    print(f"MCC:         {np.mean(mcc_scores):.4f} ± {np.std(mcc_scores):.4f}")

    #Mean Confusion Matrix 
    mean_cm = np.mean(confusion_matrices, axis=0)

    print("\nMean Confusion Matrix Across Folds")
    print("(values are averages)")
    print(pd.DataFrame(mean_cm, index=class_names, columns=class_names))
    
train_svm_from_tokens(dataset)
