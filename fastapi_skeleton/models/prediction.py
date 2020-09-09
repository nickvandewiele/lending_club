

from pydantic import BaseModel


class LoanPredictionResult(BaseModel):
    result: int
