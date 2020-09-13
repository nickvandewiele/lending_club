import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, precision_recall_curve

from src.features.build_features import build_training_features


DATA_INPUT_PATH = 'data/processed/'
MODEL_PATH = 'models/model.joblib'

def train(data):

    logger = logging.getLogger(__name__)

    # logger.info(f'Reading model features from {DATA_INPUT_PATH}...')
    # data = pd.read_csv(os.path.join(DATA_INPUT_PATH, 'training_data.csv'), parse_dates=['sec_app_earliest_cr_line'])
    y = data['target']
    X = data.drop(['target'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, random_state=0)

    cat_feat_ind = (X_train.dtypes == 'object').to_numpy().nonzero()[0]

    pool_train = Pool(X_train, y_train, cat_features=cat_feat_ind)
    pool_val = Pool(X_val, y_val, cat_features=cat_feat_ind)
    pool_test = Pool(X_test, y_test, cat_features=cat_feat_ind)

    n = y_train.value_counts().values

    logger.info('Training model...')
    model = CatBoostClassifier(learning_rate=0.03,
                            iterations=1000,
                            early_stopping_rounds=100,
                            class_weights=[1, n[0] / n[1]],
                            random_state=0)
    model.fit(pool_train, eval_set=pool_val)

    # find classification treshold that maximizes precision
    y_proba_val = model.predict_proba(pool_val)[:, 1]
    p_val, r_val, t_val = precision_recall_curve(y_val, y_proba_val)
    p_max = p_val[p_val != 1].max()
    t_all = np.insert(t_val, 0, 0)
    t_adj_val = t_all[p_val == p_max]
    y_adj_val = (y_proba_val > t_adj_val).astype(int)
    p_adj_val = precision_score(y_val, y_adj_val)
    logger.info(f'Classification threshold: {t_adj_val[0]:.3f}')
    logger.info(f'Adjusted precision (validation): {p_adj_val:.3f}')

    # metrics on test set
    y_proba_test = model.predict_proba(pool_test)[:, 1]
    y_adj_test = (y_proba_test > t_adj_val).astype(int)
    p_adj_test = precision_score(y_test, y_adj_test)
    r_adj_test = recall_score(y_test, y_adj_test)
    logger.info(f'''Adjusted precision (test): {p_adj_test:.3f}
    Adjusted recall (test): {r_adj_test:.3f}''')

    logger.info(f'Writing model to {MODEL_PATH}...')
    joblib.dump(model, open(MODEL_PATH, 'wb'))

    logger.info('Finished training model!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    data = build_training_features()
    train(data)