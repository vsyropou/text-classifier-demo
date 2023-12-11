"""
Seving model interface and base class
"""

import abc
from typing import Callable

import numpy as np
from numba import jit
from pydantic import BaseModel, ConfigDict

from app.preprocessing.label_encoder import PrimeLabelEncoder
from app.models.base import Model
from app.preprocessing.tokenization import ITokentizer


class IServingModel(abc.ABC, BaseModel):
    # TODO: Instead of arbitrary types we instead define the serialzation method
    model_config = ConfigDict(arbitrary_types_allowed=True)

    preprocessor: Callable
    tokenizer: ITokentizer
    label_encoder: PrimeLabelEncoder
    model: Model

    @abc.abstractclassmethod
    def predict(self, model_input: [str]) -> [dict[str, float]]:
        pass


class ServingModel(IServingModel):
    def predict(self, model_input: str):
        procesed = self.preprocessor([model_input])
        tokenized = self.tokenizer.tokenize(procesed)

        logits = self.model.model.predict(tokenized)

        # logits = np.exp(logits) / np.sum(np.exp(logits))

        classes = sorted(self.label_encoder.mapping.items(), key=lambda x: x[1])
        classes = map(lambda x: x[0], classes)

        intents = [(label, confidence) for label, confidence in zip(classes, logits[0])]

        intents = sorted(intents, key=lambda tpl: tpl[1], reverse=True)

        return [{"label": lbl, "confidence": cnf} for (lbl, cnf) in intents]


class BatchServingModel():
    @jit
    def patch_predict(self, model_input: [str]):
        return np.array([self.predict(x) for x in model_input])


class SoftMaxServingModel(IServingModel):
    def predict(self, model_input: str):
        procesed = self.preprocessor([model_input])
        tokenized = self.tokenizer.tokenize(procesed)

        logits = self.model.model.predict(tokenized)

        logits = np.exp(logits) / np.sum(np.exp(logits))

        classes = sorted(self.label_encoder.mapping.items(), key=lambda x: x[1])
        classes = map(lambda x: x[0], classes)

        intents = [(label, confidence) for label, confidence in zip(classes, logits[0])]

        intents = sorted(intents, key=lambda tpl: tpl[1], reverse=True)

        return [{"label": lbl, "confidence": cnf} for (lbl, cnf) in intents]
