import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Function to load data.
    args:

    messages_filepath (str): messages data filepath
    categories_filepath (str): categories data filepath
    
    Returns:

    df (pandas.DataFrame object): merged dataframe
    """
      
    
    messages = pd.read_csv(messages_filepath, sep= ',',   skiprows=0)
    categories = pd.read_csv(categories_filepath, skiprows=0)
    df = pd.merge(messages, categories, on='id', how='inner')
    return df



def clean_data(df):
    """ Function to clean data.
    args:

    df (pandas.DataFrame object): dataframe of messages and categories

    Returns:

    df (pandas.DataFrame object): cleaned dataframe
    """
    
    
    categories = pd.Series(df['categories']).str.split(pat=';', expand=True)
    row = categories.head(1)
    category_colnames = row.values.tolist()[0]
    categories.columns = category_colnames
    for column in categories:
            categories[column] = categories[column].apply(lambda x: x[-1])
            categories[column] = pd.to_numeric(categories[column])

    # Drop columns with only one unique value
    one_classifier_columns = [
        col for col in categories.columns if categories[col].nunique() < 2]
    categories = categories.drop(columns=one_classifier_columns)

    
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)
    cleaned_df = df.drop_duplicates()
    
    return cleaned_df
    
    


def save_data(df, database_filename):
    """ Saving cleaned data.
    
    args:

    df (pandas.DataFrame object): cleaned dataframe of messages and categories
    database_filepath (str): sqlite database filepath of cleaned data
    
    """
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
    print(f"database_filename: {database_filename}")

       
def main():
    if len(sys.argv) == 4:

        (messages_filepath, categories_filepath,
        database_filepath) = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
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