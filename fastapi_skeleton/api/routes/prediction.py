from fastapi import APIRouter, Depends
from starlette.requests import Request

from fastapi_skeleton.core import security
from fastapi_skeleton.models.payload import LoanPredictionPayload
from fastapi_skeleton.models.prediction import LoanPredictionResult
from fastapi_skeleton.services.models import LoanModel

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
