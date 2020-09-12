from fastapi import APIRouter, Depends
from starlette.requests import Request

from lending_club.core import security
from lending_club.models.payload import LoanPredictionPayload
from lending_club.models.prediction import LoanPredictionResult
from lending_club.services.models import LoanModel

router = APIRouter()


@router.post("/predict", response_model=LoanPredictionResult, name="predict")
def post_predict(
    request: Request,
    authenticated: bool = Depends(security.validate_request),
    block_data: LoanPredictionPayload = None
) -> LoanPredictionResult:

    model: LoanModel = request.app.state.model
    prediction: LoanPredictionResult = model.predict(block_data)

    return prediction
