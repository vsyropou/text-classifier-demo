"""
Here we keep the trainer functions. They take a model train and parse the trianing metrics
"""

import numpy as np
from dotenv import load_dotenv

from app.models.base import Model

load_dotenv()

import pandas as pd

from app.models.base import Model
from app.models.callbacks import tensorboard_callback


def train_model(
    model: Model,
    x_train: np.array,
    y_train: np.array,
    fit_args: dict,
    x_test: np.array = None,
    y_test: np.array = None,
    *,
    use_tensorboard: bool = False,
) -> (list[tuple[int | tuple[float]]], list[tuple[int | tuple[float]]], list[str]):
    test_metrics = []
    train_metrics = []

    callbacks = []
    if use_tensorboard:
        callbacks += [tensorboard_callback()]

    base_model = model.model

    history = base_model.fit(x_train, y_train, **fit_args)
    history = pd.DataFrame(history.history)

    test_history: pd.DataFrame = history[[c for c in history.columns if "val" in c]]
    train_history: pd.DataFrame = history[
        [c for c in history.columns if "val" not in c]
    ]

    test_metrics = test_history.to_records().tolist()
    train_metrics = train_history.to_records().tolist()

    return test_metrics, train_metrics, base_model.metrics_names
