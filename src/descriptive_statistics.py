import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class Descriptives:
    """
    Computes and plots descriptive statistics for a dataset.
    Useful for exploratory data .
    """
    def plot_label_distribution(self, dataset, label_column):
        """
        Plots the distribution of labels in the dataset.
        
        Args:
            dataset (pd.DataFrame): The dataset containing labels.
            label_column (str): The name of the column containing labels.In this particular case, 
            it will most likely be 'political_leaning'.
        """
        print(f"\n Plotting label distribution for column: {label_column}")
        label_counts = dataset[label_column].value_counts()
        sns.barplot(x=label_counts.index, y=label_counts.values)
        plt.title(f'Label Distribution for label {label_column}')
        plt.xlabel('Labels')
        plt.ylabel('Counts')
        plt.show()
    
    def word_count_per_label(self, dataset, text_column, label_column):
        """
        Analyzes and plots word count statistics per label.
        Args:
            dataset (pd.DataFrame): The dataset containing text and labels.
            text_column (str): The name of the column containing text data.
            label_column (str): The name of the column containing labels.
        """
        print(f"\n Below are word and character count statistics per label: {label_column}")
        dataset['char_count'] = dataset['post'].str.len()
        dataset['word_count'] = dataset['post'].str.split().str.len()

        print(dataset.groupby('political_leaning')[['char_count', 'word_count']].describe())

        sorted_wc = np.sort(dataset['word_count'])
        cdf = np.arange(len(sorted_wc)) / len(sorted_wc)

        plt.figure()
        plt.plot(sorted_wc, cdf)
        plt.xlabel("Word Count")
        plt.ylabel("Cumulative Probability")
        plt.title("CDF of Word Count")
        plt.savefig("output/word_count_cdf.png")
        plt.show()
        
    def get_author_info(self, dataset, author_column):
        """
        Computes and prints author-related statistics.
        
        Args:
            dataset (pd.DataFrame): The dataset containing author information.
            author_column (str): The name of the column containing author IDs.
        """
        print(f"\n Below are some author statistics based on column: {author_column}")
        unique_authors = dataset[author_column].nunique()
        total_posts = len(dataset)
        
        # Count posts per author
        author_counts = dataset[author_column].value_counts().reset_index()
        author_counts.columns = [author_column, 'post_count']

        # Descriptive Stats
        print(author_counts['post_count'].describe())

        # Visualize
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(author_counts['post_count'], bins=50, log_scale=(False, True))
        plt.title("Distribution of Post Counts (Log Scale)")
        plt.xlabel("Number of Posts")

        plt.subplot(1, 2, 2)
        sns.boxplot(y=author_counts['post_count'])
        plt.title("Boxplot of Post Counts")
        plt.savefig("output/author_boxplot.png")

        #Show the users with the most posts
        print("Top 10 Most Active Authors")
        print(author_counts.head(10))
        print(f"Total unique authors: {unique_authors}")
        print(f"Total posts: {total_posts}")

        # Print the percentiles, use for capping the dataset at a certain level of posts per author
        p90 = author_counts['post_count'].quantile(0.90)
        p95 = author_counts['post_count'].quantile(0.95)
        p99 = author_counts['post_count'].quantile(0.99)
        print(f"90% of users have fewer than {p90:.0f} posts.")
        print(f"95% of users have fewer than {p95:.0f} posts.")
        print(f"99% of users have fewer than {p99:.0f} posts.")

    def get_top_words_per_leaning(self, df, top_n):
        """
        Identifies and prints the top N words for each political leaning.
        Args:
            df (pd.DataFrame): The dataset containing text and labels.
            top_n (int): The number of top words to retrieve per label.
        """
        print(f"\nBelow are the top {top_n} words per political leaning:")
        # What are the most common words?
        tfidf = TfidfVectorizer(
            max_features=None,
            stop_words='english',
            ngram_range=(1,2)
        )

        X = tfidf.fit_transform(df['post'])

        feature_names = np.array(tfidf.get_feature_names_out())

        def top_tfidf_terms(label, top_n):
            """
            Helpers to get top n TF-IDF terms for a given label.
            Args: label (str): The political leaning label.
                  top_n (int): Number of top terms to return.
            """
            print(f"\nBelow are the top TF-IDF terms for label: {label}")
            # Boolean mask
            idx = df['political_leaning'] == label
            # Convert to integer row indices for sparse matrix
            row_indices = np.where(idx.to_numpy())[0]
            
            # Compute mean TF-IDF for these rows
            mean_tfidf = X[row_indices].mean(axis=0).A1
            
            # Get top n indices
            top_idx = mean_tfidf.argsort()[-top_n:][::-1]
            
            return feature_names[top_idx]

        print(f'Top words right:\n{top_tfidf_terms("right", top_n)}')
        print(f'Top words center:\n{top_tfidf_terms("center", top_n)}')
        print(f'Top words left:\n{top_tfidf_terms("left", top_n)}')