import os
import pandas as pd
import numpy as np
import joblib
import logging
import json
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier

DATA_INPUT_PATH = 'data/interim/'
DATA_MINMAX_PATH = 'data/processed/'
MODEL_PATH = 'models/model.joblib'

CLASSIFICATION_THRESHOLD = 0.999

feature_names = ["loan_amnt","mths_since_recent_inq","revol_util","bc_open_to_buy","bc_util","num_op_rev_tl","term","delinq_2yrs","sec_app_earliest_cr_line","addr_state", "emp_title"]


def predict(features):
    logger = logging.getLogger(__name__)

    logger.info(f'Reading model from {MODEL_PATH}...')
    model = joblib.load(open(MODEL_PATH, 'rb'))

    X = pd.DataFrame(data=features, columns=feature_names)
    X['mths_since_recent_inq'] = X['mths_since_recent_inq'].astype(float)
    X['sec_app_earliest_cr_line'] = pd.to_datetime(X['sec_app_earliest_cr_line'])

    # results = X[['id']]
    # X = X.drop(['id', 'loan_status'], axis=1)

    # using pandas to read dict for the automatic dtype recognition
    logger.info(f'Reading min/max from {DATA_MINMAX_PATH}...')
    with open(os.path.join(DATA_MINMAX_PATH, 'missing.json'), 'r') as fp:
        missing_r = json.load(fp)

    fill_max_values = missing_r['max']
    fill_min_values = missing_r['min']
    fill_min_values['sec_app_earliest_cr_line'] = pd.to_datetime(fill_min_values['sec_app_earliest_cr_line'])

    logger.info('Filling missing values...')
    X = X.fillna({**fill_max_values, **fill_min_values})

    logger.info('Predicting...')
    cat_feat_ind = (X.dtypes == 'object').to_numpy().nonzero()[0]
    pool = Pool(X, cat_features=cat_feat_ind)
    y_proba = model.predict_proba(pool)[:, 1]
    # results['y_pred'] = (y_proba > CLASSIFICATION_THRESHOLD).astype(int)
    results = list((y_proba > CLASSIFICATION_THRESHOLD).astype(int))

    logger.info(f'Predictions:\n{results}')
    logger.info('Finished prediction!')

    return results


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    keys = feature_names
    values =[[16000.0,"NaN",33.2,11159.0,45.8,8.0,"36 months",0.0,"NaT","CO", "driver"]]

    predict(values)