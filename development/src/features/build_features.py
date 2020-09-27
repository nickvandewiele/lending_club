import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words

# stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# tokenizer
nltk.download('punkt')

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
    "emp_title": "Employment title",
    "loan_status": "target",
}

TOP_N_MOST_FREQUENT_JOBS = 10

def build_training_features():
    logger = logging.getLogger(__name__)

    logger.info(f'Reading input data from {DATA_INPUT_PATH}...')
    data = pd.read_csv(DATA_INPUT_PATH, parse_dates=['sec_app_earliest_cr_line'])

    data = data.sample(int(1e4), random_state=42)
    
    logger.info("Preprocessing emp_title feature...")
    data.emp_title = preprocess_job_titles(data.emp_title)

    logger.info(f"Finding {TOP_N_MOST_FREQUENT_JOBS} most frequent employment titles...")
    most_frequent_titles = find_most_frequent_job_titles(data.emp_title)

    other_condition = data.emp_title.isin(most_frequent_titles)
    data.emp_title = data.emp_title.where(other_condition, 'other')
    logger.info(f"Most frequent job titles: {most_frequent_titles}")
    
    # Missing Values
    logger.info('Filling missing values...')
    fill_max = ['bc_open_to_buy', 'mths_since_recent_inq']
    fill_min = np.setdiff1d(data.columns.values, fill_max)
    data[fill_max] = data[fill_max].fillna(data[fill_max].max())
    data[fill_min] = data[fill_min].fillna(data[fill_min].min())

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
    logger.info('Finished writing model features!')

    return data


def remove_stop_words(s):
    words = word_tokenize(s)
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def convert_numbers(s):
    tokens = word_tokenize(s)
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def replace_acronyms(data):
    acronyms={
        'sr': 'senior',
        'mgr': 'manager',
        'vp': 'vice president',
        'fms': 'financial management service',
        'svp': 'senior vice president',
        'cto': 'chief technology officer',
        'cfo': 'chief financial officer',
        'coo': 'chief operating officer',
        'ceo': 'chief executive officer',
        'qa': 'quality assurance',
        'rn': 'registered nurse',
        'hr': 'human resources',
        'rep': 'representative',
        'gm': 'general manager',
        'cna': 'certified nursing assistant',
        'lpn': 'licensed practical nurse',
    }
    
    tokens = word_tokenize(data)
    
    return ' '.join([acronyms.get(i, i) for i in tokens])


def preprocess_job_titles(series):
    series = series.fillna("nan")   
    series = series.map(lambda s: s.lower())
    series = series.map(remove_stop_words)
    series = series.map(remove_punctuation)
    series = series.map(convert_numbers)
    series = series.map(replace_acronyms)
    return series


def find_most_frequent_job_titles(series):

    series = series[series != "nan"]
    occurrences = series.value_counts()

    most_frequent = occurrences.head(TOP_N_MOST_FREQUENT_JOBS).rename_axis('title').reset_index().sort_values('emp_title', ascending=False)
    most_frequent_titles = most_frequent.title.values

    return most_frequent_titles


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    build_training_features()