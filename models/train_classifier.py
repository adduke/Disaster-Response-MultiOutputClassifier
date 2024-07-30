import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

def load_data(database_filepath):
    """ Function to load data from SQLite database.
    Args:
        database_filepath (str): Filepath to the SQLite database.
  
    Returns:
        X (numpy.ndarray): Array of messages.
        Y (numpy.ndarray): Array of category labels.
        category_names (list): List of category names.
    """
    
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine).drop(columns=['id'])
    X = df['message'].values
    Y = df.drop(columns=['message', 'original', 'genre']).values
    category_names = [col for col in df.drop(columns=['message', 'original', 'genre'])]
    return X, Y, category_names

def tokenize(text):
    """ Function to tokenize and lemmatize text.
    Args:
        text (str): Input text to be tokenized.
  
    Returns:
        clean_tokens (list): List of cleaned tokens.
    """
    
    
    # Normalize text by removing punctuation and converting to lowercase
    pun_regex = r"[^a-zA-Z0-9]"
    text = re.sub(pun_regex, " ", text)
    
    # Remove URLs from the text
    url_regex = (
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = re.sub(url_regex, " ", text)

    # Tokenize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize and clean tokens
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

def build_model():
    """ Function to build a machine learning pipeline.
    Returns:
        pipeline (Pipeline): Scikit-learn Pipeline object.
    """
    
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression()))
    ])
   
    # Set hyperparameters for the pipeline
    parameters = {
    #removing very common words (e.g., "the", "and") may be useful
    'vect__max_df': [0.75, 1.0],
    # Chose not to optimise by rare terms
    # Chose not to alter by reducing the count of tokens
    # Optimise regularisation strength for overfitting
    'clf__estimator__C': [0.01, 0.1, 1.0, 10.0],
    # Different can affect convergence speed and model performance
    'clf__estimator__solver': ['liblinear', 'saga']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, y_test, category_names, show_plot=False):
    """ Function to evaluate the model's performance.
    Args:
        model (Pipeline): Trained model.
        X_test (numpy.ndarray): Test data features.
        y_test (numpy.ndarray): Test data labels.
        category_names (list): List of category names.
        show_plot (bool): Flag to display confusion matrix heatmaps.
    """
    
    
    y_pred = model.predict(X_test)
    for i in range(y_test.shape[1]):
        category_name = category_names[i]
        labels = np.unique(y_pred[:, 1])
        cf_matrix = confusion_matrix(y_test[:, i], y_pred[:, i], labels=labels)
        accuracy = (y_pred[:, i] == y_test[:, i]).mean()

        # Print classification report and accuracy
        print('\033[91m' + f'{category_name}' + '\033[90m')
        print(classification_report(y_test[:, i], y_pred[:, i], labels=labels))
        if show_plot:
            # Plot confusion matrix heatmap
            import seaborn as sns
            import matplotlib.pyplot as plt
            plt.figure(figsize=(4, 3))  
            sns.heatmap(cf_matrix, annot=True, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.title(f'Heatmap {category_name}')
            plt.show()
        print("Accuracy:", accuracy)
        print()

def save_model(model, model_filepath):
    """ Function to save the trained model to a file.
    Args:
        model (Pipeline): Trained model.
        model_filepath (str): Filepath to save the model.
    """
    
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Main entry point for running the machine learning pipeline.

    This function:
    1. Checks if the correct number of command-line arguments is provided.
    2. Loads data from the specified database file.
    3. Splits the data into training and test sets.
    4. Builds a machine learning model pipeline.
    5. Trains the model using the training data.
    6. Evaluates the model's performance on the test data.
    7. Saves the trained model to the specified file.

    Command-line Arguments:
    - database_filepath (str): Path to the SQLite database file containing the cleaned data.
    - model_filepath (str): Path where the trained model will be saved.

    Example usage:
    python train_classifier.py data/DisasterResponse.db models/classifier.pkl

    Outputs:
    - Loads and prints information about the database file.
    - Loads and splits the data.
    - Builds, trains, and evaluates the model, printing evaluation results.
    - Saves the trained model to the specified file path.

    Notes:
    - If the number of arguments is incorrect, an error message is printed with instructions.
    """
        
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
    else:
        print('Please provide the filepaths of the database and the model '\
              'as the first and second argument respectively. \n\nExample: '\
              'python train_classifier.py database_filepath model_filepath')

if __name__ == '__main__':
    main()
