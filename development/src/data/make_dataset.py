# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd

from src.features.build_features import FEATURES

start_date = '2018-01-01 00:00:00'
end_date = '2019-01-01 00:00:00'


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    logger.info(f'Reading raw data from {input_filepath}...')
    data = pd.read_csv(
        input_filepath,
        parse_dates=['issue_d'],
        infer_datetime_format=True,
        low_memory=False,
    )

    logger.info(f'Only keeping data from {start_date} until {end_date}...')
    data = data[(data.issue_d >= start_date) & (data.issue_d < end_date)]
    data = data.reset_index(drop=True)

    # logger.info('Reading model input features...')
    # features = pd.read_csv('features/features.txt', 
    #     names=['features', 'dtype', 'description'], 
    #     header=None)

    logger.info(f'Only keeping features {FEATURES}...')
    # X = data[features.features.values]
    X = data[FEATURES.keys()]

    logger.info(f'Writing processed data to {output_filepath}...')
    X.to_csv(output_filepath, index=False)

    logger.info('Finished writing processed data!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
