# FastAPI Model Server Skeleton

Serving machine learning models production-ready, fast, easy and secure powered by the great FastAPI by [Sebastián Ramírez]([)](https://github.com/tiangolo).

## Requirements

Python 3.6+

## Installation
Install the required packages in your local environment (ideally virtualenv, conda, etc.).
```bash
pip install -r requirements
``` 


## Setup
1. Duplicate the `.env.example` file and rename it to `.env` 


2. In the `.env` file configure the `API_KEY` entry. The key is used for authenticating our API. <br>
   A sample API key can be generated using Python REPL:
```python
import uuid
print(str(uuid.uuid4()))
```

## Run It

1. Start your  app with: 
```bash
uvicorn app.main:app
```

2. Go to [http://localhost:8000/docs](http://localhost:8000/docs).
   
3. Click `Authorize` and enter the API key as created in the Setup step.
![Authroization](./docs/authorize.png)
   
4. You can use the sample payload from the `docs/loan_payload.json` file when trying out the model using the API.
   ![Prediction with example payload](./docs/loan_payload.png)

## Run Tests

If you're not using `tox`, please install with:
```bash
pip install tox
```

Run your tests with: 
```bash
tox
```

This runs tests and coverage for Python 3.6 and Flake8, Autopep8, Bandit.
