from app.core import config


def test_prediction(test_client) -> None:
    response = test_client.post(
        "/api/model/predict",
        json={
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
        },
        headers={"token": str(config.API_KEY)},
    )
    assert response.status_code == 200
    assert "result" in response.json()


def test_prediction_nopayload(test_client) -> None:
    response = test_client.post(
        "/api/model/predict", json={}, headers={"token": str(config.API_KEY)}
    )
    assert response.status_code == 422
