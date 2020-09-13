import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import json

DATA_INPUT_PATH = 'data/interim/input_data.csv'
DATA_OUTPUT_PATH = 'data/processed/'

FEATURES = {
    "loan_amnt": "The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.",
    "mths_since_recent_inq": "Months since oldest bank installment account opened",
    "revol_util": "Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.",
    "bc_open_to_buy": "Total open to buy on revolving bankcards.",
    "bc_util": "Ratio of total current balance to high credit/credit limit for all bankcard accounts.",
    "num_op_rev_tl": "Number of open revolving accounts",
    "term": "The number of payments on the loan. Values are in months and can be either 36 or 60.",
    "delinq_2yrs": "The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years",
    "sec_app_earliest_cr_line": "Earliest credit line at time of application for the secondary applicant",
    "addr_state": "The state provided by the borrower in the loan application",
    "loan_status": "target",
}

def build_training_features():
    logger = logging.getLogger(__name__)

    logger.info(f'Reading input data from {DATA_INPUT_PATH}...')
    data = pd.read_csv(DATA_INPUT_PATH, parse_dates=['sec_app_earliest_cr_line'])

    # Missing Values
    logger.info('Filling missing values...')
    fill_max = ['bc_open_to_buy', 'mths_since_recent_inq']
    fill_min = np.setdiff1d(data.columns.values, fill_max)
    data[fill_max] = data[fill_max].fillna(data[fill_max].max())
    data[fill_min] = data[fill_min].fillna(data[fill_min].min())

    logger.info(f'Writing feature max/min to {DATA_OUTPUT_PATH}...')
    # data[fill_max].max().reset_index().T.to_csv(os.path.join(DATA_OUTPUT_PATH, 'max.csv'), header=False, index=False)
    # data[fill_min].min().reset_index().T.to_csv(os.path.join(DATA_OUTPUT_PATH, 'min.csv'), header=False, index=False)

    missing = {}
    missing['max'] = data[fill_max].max().to_dict()
    missing['min'] = data[fill_min].min().to_dict()

    with open(os.path.join(DATA_OUTPUT_PATH, 'missing.json'), 'w') as fp:
        json.dump(missing, fp, default=str, indent=4)

    # Target
    y = data['loan_status'].copy()
    y = y.isin(['Current', 'Fully Paid', 'In Grace Period']).astype('int')
    data['target'] = y
    data = data.drop(['loan_status'], axis=1)

    logger.info(f'Writing features to {DATA_OUTPUT_PATH}...')
    data.to_csv(os.path.join(DATA_OUTPUT_PATH, 'training_data.csv'), index=False)

    return data

    logger.info('Finished writing model features!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    build_training_features()