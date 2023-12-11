"""
Here we keep funcitons to aid the api with serving the model and keep its code clean

"""

import os
from functools import partial
from typing import Any

import cloudpickle
from dotenv import load_dotenv
from fastapi.exceptions import HTTPException

from app.models.serving import ServingModel

load_dotenv()

MODEL_PATH = os.environ["SERVE_MODEL_PATH"]
MODEL_TAGS = os.environ["SERVE_MODEL_TAGS"]

# TODO: adopt pydantic exception response schema


def _load_with_exception(path: str) -> ServingModel | dict:
    """Loads something from path using cloudpickle"""
    try:
        return cloudpickle.load(open(path, "rb"))
    except Exception as err:
        detail = {
            "label": "INTERNAL_ERROR",
            "message": f"Failed to load model; This should not happen {err.args} ",
        }
        raise HTTPException(status_code=500, detail=detail) from err


def _predict_with_exception(
    model: ServingModel, model_input: str, top_n: int = 3
) -> float:
    """
    Predict "wrapper" that throws an http exeption in case of an error

    Args:
        model: Any instanse of a model that has a predict function
        model_input: parsed input request in the shape of IntentRequest
    Returns:
        model prediction

    """
    try:
        model_output = model.predict(model_input)[:top_n]
    except Exception as err:
        detail = {
            "label": "INTERNAL_ERROR",
            "message": f"Failed to evaluate model; This should not happen; {err.args} ",
        }
        raise HTTPException(500, detail=detail) from err

    return model_output


def test_model(model: ServingModel):
    try:
        model(model_input="what is my intent")
    except Exception as err:
        detail = {
            "label": "INTERNAL_ERROR",
            "message": f"Loaded model fails to evaluate; This should not happen; {err.args} ",
        }
        raise HTTPException(status_code=500, detail="") from err


def load_model(top_n: int) -> ServingModel:
    """
    Helper generator function to facilitate the model dependency in the input request

    Returns a callable that takes the request payload as input and returns the model's prediction.
    """
    model = _load_with_exception(MODEL_PATH)

    assert model is not None, "Model is None"

    assert hasattr(model, "predict"), f"Model {model} has no predict method"

    model_callable = partial(_predict_with_exception, model=model, top_n=top_n)

    test_model(model_callable)

    return model_callable


def load_tags() -> dict:
    """
    Helper generator function to facilitate the dependency of the model tags

    """
    return _load_with_exception(MODEL_TAGS) or None
