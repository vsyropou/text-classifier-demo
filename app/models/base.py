"""
Interfaces of the model and model builder to be used during training 
"""

import abc
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict


class ITrainingHistory(abc.ABC):
    @abc.abstractproperty
    def history(self) -> dict[str, (int | float)]:
        """A dictionary containing the metrics per epoch"""
        pass

    @abc.abstractproperty
    def columns(self):
        """Columns refer to the metric names that were used during training"""
        pass


class IModel(abc.ABC):
    @abc.abstractclassmethod
    def fit(self, *args, **kwargs) -> ITrainingHistory:
        pass

    @abc.abstractproperty
    def history(self):
        pass

    @abc.abstractproperty
    def metrics_names(self) -> [str]:
        pass


class Model(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    base_model: Any

    @property
    def model(self) -> IModel:
        return self.base_model


class IModelBuilder(abc.ABC):
    @abc.abstractclassmethod
    def build(self, embeddings_matrix: np.array = None) -> Model:
        pass
