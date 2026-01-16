# Political Leaning Classification

This project aims to classify political leaning based on text data. It utilizes a Support Vector Machine (SVM) model to predict whether the author of a text has a left, right, or center political leaning, as well as BERT. The project is designed to be a reproducible research pipeline, including data preprocessing, model training, and evaluation.

## tl;dr

This repository contains Python code to train and evaluate an SVM model for political leaning classification. The main script `main.py` runs the experiment based on a `config.json` file, which allows for easy manipulation of the experiment.

## Citation

If you use this code in your research, please cite:

```
@misc{your_name_2024_languageandai,
  author = {Raluca Marzac, Joris Lesterhuis, Sven Collins, Sofiia Larina},
  title = {Political Leaning Classification},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/UberGum123/language-and-ai}},
}
```

## Reproduction Instructions

### System

The code was developed and tested on Windows 10 with Python 3.12.

### Dependencies

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

The dependencies are:
```
pandas==1.4.4
matplotlib==3.4.3
jupyter==1.0.0
numpy==1.20.3
sklearn==1.8.0
nltk==3.8.1
contractions==0.1.73
pandarallel==1.6.5
seaborn ==  0.13.2
transformers
accelerator
seaborn

```

### Data

The expected dataset is a CSV file with columns for text, author, and political leaning. The path to the data and the column names can be specified in `config.json`. For this project, the data is expected at `data/political_leaning.csv`.

### Running the Experiment

To run the experiment, execute the `main.py` script:

```bash
python main.py
```

The script will create `cache` and `output` directories if they do not exist. The results of the experiment, including logs and plots, will be saved in the `output` directory.

## Experimental Manipulation

The experiment can be configured by modifying the `config.json` file. The following parameters can be changed:

- **`data`**:
    - `file_path`: Path to the dataset.
    - `text_column`: Name of the column containing the text.
    - `label_column`: Name of the column containing the labels.
    - `author_column`: Name of the column containing author IDs.
- **`descriptive_statistics`**:
    - `enabled`: Set to `true` to run descriptive statistics.
    - `plot_label_distribution`: Set to `true` to plot the distribution of labels.
    - `word_count_per_label`: Set to `true` to get word counts per label.
    - `get_author_info`: Set to `true` to get information about authors.
    - `get_top_n_words_per_label`: Number of top words to retrieve for each label.
- **`modeling`**:
    - `enabled`: Set to `true` to run the modeling part.
    - `model_type`: The type of model to use ("SVM" or "BERT").
    - `hyperparameters`:
        - `C`: A list of C values to test for the SVM model.
        - For BERT: [batch_size, epochs, max_length, learning_rate, model_type]
- **`preprocessing`**:
    - `number_of_folds`: Number of folds for cross-validation.
- **`masking`**:
    - `enabled`: Set to `true` to enable masking.
    - `masking_strategy`: The masking strategy to use.
- **`load_dataset`**: set to true to load a pre-processed dataset from cache (saves time!)
## Project Structure

- **`main.py`**: The main entry point to run the experiment.
- **`config.json`**: Configuration file for the experiment.
- **`requirements.txt`**: A list of all the packages used in the project.
- **`src/`**: Contains the source code for the project.
    - **`experiment_environment.py`**: The main class that orchestrates the experiment.
    - **`reader.py`**: Reads and preprocesses the data.
    - **`model.py`**: Implements the SVM model.
    - **`descriptive_statistics.py`**: Generates descriptive statistics.
    - **`visualizations.py`**: Creates plots and visualizations.
- **`data/`**: Should contain the dataset.
- **`output/`**: Contains the output of the experiment (logs, plots).
- **`cache/`**: Caches intermediate results.

## How to Extend

The project is modular and can be extended. For example, to add a new model, you would need to:

1.  Add the model implementation in `src/model.py`.
2.  Add a new model type in `src/experiment_environment.py` to select your new model.
3.  Add the new model and its parameters to `config.json`.
 `LICENSE` file has not been added yet, but MIT is a common choice for open source projects.)*
