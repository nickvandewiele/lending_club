
import pytest

from lending_club.core import config
from lending_club.models.payload import LoanPredictionPayload
from lending_club.models.prediction import LoanPredictionResult
from lending_club.services.models import LoanModel


def test_prediction(test_client) -> None:
    model_path = config.DEFAULT_MODEL_PATH
    minmax_path = config.DEFAULT_MINMAX_PATH
    lpp = LoanPredictionPayload.parse_obj({
        "loan_amnt": 16000.0,
        "mths_since_recent_inq": 25.0,
        "revol_util": 33.2,
        "bc_open_to_buy": 11159.0,
        "bc_util": 45.8,
        "num_op_rev_tl": 8.0,
        "term": "36 months",
        "delinq_2yrs": 0.0,
        "sec_app_earliest_cr_line": "NaT",
        "addr_state": "CO",
    })

    lpm = LoanModel(model_path, minmax_path)
    result = lpm.predict(lpp)
    assert isinstance(result, LoanPredictionResult)


def test_prediction_no_payload(test_client) -> None:
    model_path = config.DEFAULT_MODEL_PATH
    minmax_path = config.DEFAULT_MINMAX_PATH
    
    lpm = LoanModel(model_path, minmax_path)
    with pytest.raises(ValueError):
        result = lpm.predict(None)
        assert isinstance(result, LoanPredictionResult)
