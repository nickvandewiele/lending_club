## Model Description

### Lending Club 

#### Model Features
- loan_amnt: The listed amount of the loan applied for by the borrower.
- mths_since_recent_inq: Months since oldest bank installment account opened
- revol_util: Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit
- bc_open_to_buy: Total open to buy on revolving bankcards
- bc_util: Ratio of total current balance to high credit/credit limit for all bankcard accounts
- num_op_rev_tl: Number of open revolving accounts
- term: The number of payments on the loan. Values are in months and can be either 36 or 60
- delinq_2yrs: The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
- sec_app_earliest_cr_line: Earliest credit line at time of application for the secondary applicant
- addr_state: The state provided by the borrower in the loan application

#### Target variable
- Loan is good (1) or bad (0)

### References
[1] https://www.kaggle.com/pavlofesenko/minimizing-risks-for-loan-investments#5.-Model-adjustment<br>