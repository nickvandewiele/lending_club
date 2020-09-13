from fastapi import APIRouter, Depends
from starlette.requests import Request

from app.core import security
from app.models.payload import LoanPredictionPayload
from app.models.prediction import LoanPredictionResult
from app.services.models import LoanModel

router = APIRouter()


@router.post("/predict", response_model=LoanPredictionResult, name="predict")
def post_predict(
    request: Request,
    authenticated: bool = Depends(security.validate_request),
    block_data: LoanPredictionPayload = None,
) -> LoanPredictionResult:

    model: LoanModel = request.app.state.model
    prediction: LoanPredictionResult = model.predict(block_data)

    return prediction
