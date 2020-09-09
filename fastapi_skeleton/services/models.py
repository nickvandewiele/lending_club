

from typing import List

import json
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from catboost import Pool

from fastapi_skeleton.core.messages import NO_VALID_PAYLOAD
from fastapi_skeleton.models.payload import (LoanPredictionPayload,
                                             payload_to_list)
from fastapi_skeleton.models.prediction import LoanPredictionResult


class LoanModel(object):

    feature_names = ["loan_amnt","mths_since_recent_inq","revol_util","bc_open_to_buy","bc_util","num_op_rev_tl","term","delinq_2yrs","sec_app_earliest_cr_line","addr_state"]

    def __init__(self, path, minmax_path):
        self.path = path
        self.minmax_path = minmax_path
        self._load_local_model()
        self._load_local_minmax()

        self.threshold = 0.999

    def _load_local_model(self):
        self.model = joblib.load(self.path)

    def _load_local_minmax(self):
        with open(self.minmax_path, 'r') as fp:
            self.minmax = json.load(fp)        

    def _pre_process(self, payload: LoanPredictionPayload) -> List:
        logger.debug("Pre-processing payload.")
        result = [payload_to_list(payload)]
        return result

    def _post_process(self, prediction: np.ndarray) -> LoanPredictionResult:
        logger.debug("Post-processing prediction.")
        result = prediction.tolist()[0]
        lpp = LoanPredictionResult(result=result)
        return lpp

    def _predict(self, features: List) -> np.ndarray:
        logger.debug("Predicting.")
        
        X = pd.DataFrame(data=features, columns=self.feature_names)
        X['mths_since_recent_inq'] = X['mths_since_recent_inq'].astype(float)
        X['sec_app_earliest_cr_line'] = pd.to_datetime(X['sec_app_earliest_cr_line'])

        logger.info('Filling missing values...')
        fill_max_values = self.minmax['max']
        fill_min_values = self.minmax['min']
        fill_min_values['sec_app_earliest_cr_line'] = pd.to_datetime(fill_min_values['sec_app_earliest_cr_line'])        
        X = X.fillna({**fill_max_values, **fill_min_values})

        logger.info('Predicting...')
        cat_feat_ind = (X.dtypes == 'object').to_numpy().nonzero()[0]
        y_proba = self.model.predict_proba(Pool(X, cat_features=cat_feat_ind))[:, 1]
        prediction_result = (y_proba > self.threshold).astype(int)

        return prediction_result

    def predict(self, payload: LoanPredictionPayload):
        if payload is None:
            raise ValueError(NO_VALID_PAYLOAD.format(payload))

        pre_processed_payload = self._pre_process(payload)
        prediction = self._predict(pre_processed_payload)
        logger.info(prediction)
        post_processed_result = self._post_process(prediction)

        return post_processed_result
