

from typing import Callable

from fastapi import FastAPI
from loguru import logger

from lending_club.core.config import DEFAULT_MODEL_PATH, DEFAULT_MINMAX_PATH
from lending_club.services.models import LoanModel


def _startup_model(app: FastAPI) -> None:
    model_path = DEFAULT_MODEL_PATH
    minmax_path = DEFAULT_MINMAX_PATH
    model_instance = LoanModel(model_path, minmax_path)
    app.state.model = model_instance


def _shutdown_model(app: FastAPI) -> None:
    app.state.model = None


def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        logger.info("Running app start handler.")
        _startup_model(app)
    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        logger.info("Running app shutdown handler.")
        _shutdown_model(app)
    return shutdown
