
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
matplotlib.use('Agg')  # Use a non-interactive backend for environments without display

class Visualizer:
    """Handles all visualization tasks for the Modeler."""
    
    def plot_confusion_matrix(self, cm, class_names, title='Confusion Matrix'):
        """
        Plots the confusion matrix for the given true and predicted labels.
        Args:
            y_true (list): True labels.
            y_pred (list): Predicted labels.
            class_names (list): List of class names for labeling the axes.
        """

        plt.figure(figsize=(8, 6))
        
        # heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names
        )
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"output/{title}.png")
        plt.show()
        
    def plot_metrics_summary(self, metrics_dict, title):
        """
        Plots a bar chart with error bars for different metrics (Macro F1, Weighted F1, MCC).
        
        Args:
            metrics_dict (dict): Dictionary where keys are metric names (str) 
                                and values are lists of scores (one per fold).
                                Example: {'Macro F1': [0.5, 0.6], 'MCC': [0.4, 0.5]}
        """
        # Prepare data for Seaborn
        data = []
        for metric_name, scores in metrics_dict.items():
            for score in scores:
                data.append({'Metric': metric_name, 'Score': score})
        
        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        
        sns.barplot(x='Metric', y='Score', data=df, errorbar='sd', capsize=.1, palette="viridis")
        
        plt.title(title)
        plt.ylim(0, 1.05)  
        plt.tight_layout()
        plt.savefig(f"output/{title}.png")
    
    def plot_top_features(self, vectorizer, model, class_names, top_n, title):
        """
        Plots the most positive feature weights (n-grams) per class.
        
        Args:
            vectorizer: The fitted TfidfVectorizer.
            model: The fitted LinearSVC model.
            class_names: List of class names.
            top_n: Number of top features to show.
        """
        feature_names = vectorizer.get_feature_names_out()
                
        num_classes = len(class_names)
        fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 6), sharey=False)
        
        if num_classes == 1: axes = [axes] # Handle single plot case
        
        for i, class_label in enumerate(class_names):
            # Get coefficients for this class
            if num_classes > 2:
                coefs = model.coef_[i]
            else:
                coefs = model.coef_[i]

            # Sort indices
            top_indices = np.argsort(coefs)[-top_n:] # Top N positive
            
            top_features = [feature_names[j] for j in top_indices]
            top_weights = coefs[top_indices]
            
            # Plot
            ax = axes[i] if num_classes > 1 else axes
            
            # Horizontal bar plot
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_weights, align='center', color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features)
            ax.set_title(f"Top {top_n} Features: {class_label}")
            ax.set_xlabel("Coefficient Weight")

        plt.tight_layout()
        plt.savefig(f"output/{title}.png")
        
    def plot_pca_embeddings(self, X_vec, y, class_names, title="PCA of Document Vectors"):
        """
        Reduces TF-IDF vectors to 2D using TruncatedSVD (works with sparse matrices).
        """
    
        print(f"Reducing {X_vec.shape[0]} documents from {X_vec.shape[1]} to 2 dimensions...")
        
        svd = TruncatedSVD(n_components=2, random_state=42)
        X_reduced = svd.fit_transform(X_vec)
        
        plt.figure(figsize=(10, 8))
        
        # Create a DataFrame for easy plotting with Seaborn
        df_plot = pd.DataFrame(data=X_reduced, columns=['Component 1', 'Component 2'])
        
        # If y is encoded (0,1,2), map to names
        if len(y) > 0 and np.issubdtype(type(y[0]), np.integer):
            df_plot['Label'] = [class_names[i] for i in y]
        else:
            df_plot['Label'] = y

        sns.scatterplot(
            x='Component 1', 
            y='Component 2', 
            hue='Label', 
            style='Label',
            data=df_plot, 
            palette='deep',
            s=60,
            alpha=0.8
        )
        
        # Add explained variance to title
        explained_var = svd.explained_variance_ratio_.sum() * 100
        plt.title(f"{title}\n({explained_var:.1f}% variance explained)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(f"output/{title}.png")
        plt.show()