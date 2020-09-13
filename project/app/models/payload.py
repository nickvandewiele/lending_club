from typing import List
from pydantic import BaseModel


class LoanPredictionPayload(BaseModel):

    loan_amnt: float
    mths_since_recent_inq: float
    revol_util: float
    bc_open_to_buy: float
    bc_util: float
    num_op_rev_tl: float
    term: str
    delinq_2yrs: float
    sec_app_earliest_cr_line: str
    addr_state: str


def payload_to_list(lpp: LoanPredictionPayload) -> List:
    return [
        lpp.loan_amnt,
        lpp.mths_since_recent_inq,
        lpp.revol_util,
        lpp.bc_open_to_buy,
        lpp.bc_util,
        lpp.num_op_rev_tl,
        lpp.term,
        lpp.delinq_2yrs,
        lpp.sec_app_earliest_cr_line,
        lpp.addr_state,
    ]
