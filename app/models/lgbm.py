import logging

import numpy as np
from lightgbm import LGBMRegressor
from pydantic import BaseModel

from app.models.base import IModelBuilder

# NOTE: This is work in progress. Dont consider this as finished


class LGBMRegressor(IModelBuilder, BaseModel):
    input_dim: int
    input_length: int
    n_target_classes: int

    def build(
        self,
    ) -> LGBMRegressor:
        logging.warn("LGBMRegressor is work in progress some features might not work")
        # Parameters we"ll use for the prediRction
        parameters = {
            "n_estimators": 1000,
            "early_stopping_rounds": 20,
            "application": "binary",
            "objective": "binary",
            "metric": "auc",
            "boosting": "dart",
            "num_leaves": 31,
            "feature_fraction": 0.5,
            "bagging_fraction": 0.5,
            "bagging_freq": 20,
            "learning_rate": 0.05,
            "verbose": 0,
        }
        self.model = LGBMRegressor(**parameters)

    def fit(self, x_train: np.array, y_train: np.array, weights: np.array = None):
        logging.warn("LGBMRegressor is work in progress some features might not work")
        args = [x_train, y_train]
        if weights:
            args += [weights]
        self.model.fit(x_train, y_train)
