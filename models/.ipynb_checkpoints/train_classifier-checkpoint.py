import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import f1_score, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import pickle


def load_data(database_filepath):
    """ Function to load data.
    args:
    database_filepath (str): sqlite database filepath of cleaned data
  
    Returns:

    X (pandas.DataFrame object): messages dataframe
    Y (pandas.DataFrame object): categories MultiOutput target
    category_names (list): List of category names 
    
    """
    
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine).drop(columns=['id'])
    X = df['message'].values
    Y = df.drop(columns=['message', 'original', 'genre']).values
    category_names = [col[:-2] for col in df.drop(
        columns=['message', 'original', 'genre'])]
    print(database_filepath)
    return X, Y, category_names


def tokenize(text):
    """ Function to tokenize messages
    args:
    text (str): message text string
  
    Returns:

    clean_tokens (array): tokenized text array
    
    """
    
    
    # convert all into lower case
    # Normaize data - remove dashes, puctuation etc.
    pun_regex = r"[^a-zA-Z0-9]"
    #text = text.apply(lambda x: re.sub(pun_regex, " ", x))
    text = re.sub(pun_regex, " ", text)
    # Remove all urls users may have left
    url_regex = (
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]'
    r'|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    #text = text.apply(lambda x: re.sub(url_regex, '', x))
    text = re.sub(url_regex, " ", text)

    #tokens = text.apply(lambda x: word_tokenize(x))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    
    """ Initiate fitting model.
    Returns:
    
    Pipeline (pipeline object): combined transformer and fitting model
   
    """
        
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression()))
        ])
   
    parameters = {'clf__estimator__C': 10.0,
                  'clf__estimator__max_iter': 200,
                  'clf__estimator__solver': 'liblinear',
                  'vect__max_df': 0.75}

    pipeline.set_params(**parameters)
    return pipeline

def evaluate_model(model, X_test, y_test, category_names, show_plot=False):
    
    """ Precision, recall, F1 score, confusion matrix and accuracy summary.
    args:
    
    model (object): combined transformer and fitting model
    X_test (pandas.DataFrame object): test dataframe for validation
    y_test (pandas.DataFrame object): test target for validation
    category_names (list): List of category names 
    
    """
        
    y_pred = model.predict(X_test)
    for i in range(y_test.shape[1]):
        category_name = category_names[i]
        labels = np.unique(y_pred[:, 1])
        cf_matrix = confusion_matrix(
            y_test[:, i], y_pred[:, i], labels=labels)
        accuracy = (y_pred[:, i] == y_test[:, i]).mean()

        # Creating a report for precision, recall and f1 score
        print('\033[91m' + f'{category_name}' + '\033[90m')
        print(classification_report(y_test[:,i], y_pred[:,i], labels=labels))
        if show_plot:
            # Creating a seaborn confusion Matrix
            plt.figure(figsize=(4, 3))  
            # Create the heatmap
            sns.heatmap(
                cf_matrix, annot=True, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
            # Add a title for each heatmap (optional)
            plt.title(f'Heatmap {category_name}')
            plt.show()
        else:
            pass
        print("Accuracy:", accuracy)
        print()


def save_model(model, model_filepath):
    
    """ Precision, recall, F1 score, confusion matrix and accuracy summary.
    args:
    
    model (object): combined transformer and fitting model
    model_filepath (str): file path for pkl file to be saved
    
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)
        
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
        print(
          'Please provide the filepaths of the messages and categories '\
          'datasets as the first and second argument respectively, as '\
          'well as the filepath of the database to save the cleaned data '\
          'to as the third argument. \n\nExample: python process_data.py '\
          'disaster_messages.csv disaster_categories.csv '\
          'InsertDatabaseName.db')


if __name__ == '__main__':
    main()