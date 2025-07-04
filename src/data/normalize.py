import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.model_selection import train_test_split
from check_structure import check_existing_file, check_existing_folder
import os
from sklearn.preprocessing import MinMaxScaler

@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn preprocessed data from (../processed) into
        cleaned data ready to be analyzed (saved in../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    input_filepath = click.prompt('Enter the file path for the input data (data/processed)', type=click.Path(exists=True))
    input_filepath_train = f"{input_filepath}/X_train.csv"
    input_filepath_test = f"{input_filepath}/X_test.csv"
    output_filepath = click.prompt('Enter the file path for the output preprocessed data (data/processed)', type=click.Path())

    process_data(input_filepath_train, input_filepath_test, output_filepath)

def process_data(input_filepath_train, input_filepath_test, output_filepath):
    # Import datasets
    df_train = import_dataset(input_filepath_train, sep=",")
    df_test = import_dataset(input_filepath_test, sep=",")

    # Normalize
    df_train = normalize(df_train)
    df_test = normalize(df_test)
 
    # Create folder if necessary
    create_folder_if_necessary(output_filepath)

    # Save dataframes
    save_dataframes(df_train, 'X_train_scaled', output_filepath)
    save_dataframes(df_test, 'X_test_scaled', output_filepath)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)


def normalize(df):
    # Normalize
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)

def save_dataframes(df, filename, output_folderpath):
    # Save dataframe
    output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
    df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()