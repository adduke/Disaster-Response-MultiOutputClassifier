import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Function to load data from CSV files and merge them.
    Args:
        messages_filepath (str): Filepath for messages data CSV.
        categories_filepath (str): Filepath for categories data CSV.
    
    Returns:
        df (pandas.DataFrame): Merged dataframe containing messages and categories.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath, sep=',', skiprows=0)
    # Load categories dataset
    categories = pd.read_csv(categories_filepath, skiprows=0)
    # Merge datasets on 'id'
    df = pd.merge(messages, categories, on='id', how='inner')
    return df

def clean_data(df):
    """ Function to clean the merged dataframe.
    Args:
        df (pandas.DataFrame): Dataframe containing merged messages and categories.
    
    Returns:
        df (pandas.DataFrame): Cleaned dataframe with categories split into separate columns.
    """
    # Split the categories column into separate category columns
    categories = pd.Series(df['categories']).str.split(pat=';', expand=True)
    # Extract the first row to get category names
    row = categories.head(1)
    category_colnames = row.values.tolist()[0]
    # Rename columns of the categories dataframe with clearly named columns

    categories.columns = [col[:-2] for col in category_colnames]

    # Convert category values to 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])  # Keep only the last character
        categories[column] = pd.to_numeric(categories[column])  # Convert to numeric type

    
    # Drop rows where classification is not binary, strictly keep 0s and 1s only
    remove_anomalies = []
    for col in categories.columns:
        remove_anomalies.append(list(categories[categories[col] == 2].index))
    categories = categories.drop(index=remove_anomalies[0])
    assert categories.max().unique()[0] == 1, "classes exist greater than 1"
    assert categories.min().unique()[0] == 0, "classes exist below 0"
    
    # Drop columns with only one unique value (not useful for classification)
    one_classifier_columns = [col for col in categories.columns if categories[col].nunique() < 2]
    categories = categories.drop(columns=one_classifier_columns)
    assert categories.nunique().value_counts().shape  == (1,), (
    "A column in categories exists without 2 classes")

    # Drop the original categories column from the dataframe
    df = df.drop(columns=['categories'])
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # Remove duplicates
    df = df.drop_duplicates()
    
    assert len(df[df.duplicated()]) == 0

    # drop messages with nulls as we need dataset to be complete for prediction
    df = df.dropna(subset=['message','original'])

    # drop messages with nulls from categories
    cleaned_df = df.dropna()
    assert cleaned_df.isna().sum().max() == 0, (
        "there are still entries with missing data")

    
    
    
    
    return cleaned_df

def save_data(df, database_filename):
    """ Function to save cleaned data into a SQLite database.
    Args:
        df (pandas.DataFrame): Cleaned dataframe containing messages and categories.
        database_filename (str): Filepath for the SQLite database.
    """
    # Create a SQLite database engine
    engine = create_engine(f'sqlite:///{database_filename}')
    # Save the dataframe to a table named 'DisasterResponse'
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
    print(f"database_filename: {database_filename}")

def main():
    if len(sys.argv) == 4:
        # Get filepaths from command line arguments
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        # Load data from the filepaths
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        # Clean the loaded data
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        # Save the cleaned data to the specified SQLite database
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'InsertDatabaseName.db')

if __name__ == '__main__':
    main()
