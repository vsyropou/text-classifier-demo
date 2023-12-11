"""
Here we define tokenizer objects for text sequences
"""

import abc
import logging
from typing import Iterable

import numpy as np
from keras.preprocessing.sequence import pad_sequences as keras_pad_sequences
from keras.preprocessing.text import Tokenizer as _KerasTokenizer
from pydantic import BaseModel, ConfigDict, Field


class ITokentizer(abc.ABC, BaseModel):
    # TODO: Instead of arbitrary types we should instead define the serialzation method
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_sequence_lengh: int
    word_index: dict[str, int] = None
    vocabulary_size: int = None

    @abc.abstractclassmethod
    def fit(self, data: Iterable[str]) -> None:
        pass

    @abc.abstractclassmethod
    def tokenize(self, data: Iterable[str]) -> np.array:
        pass


class KerasTokenizer(ITokentizer, BaseModel):
    """
    Just a a wrapper arround
        `'keras.preprocessing.text.Tokenizer'
        'keras.preprocessing.sequence.pad_sequences'
    objects
    """

    tokenizer: _KerasTokenizer = Field(default_factory=lambda: _KerasTokenizer())

    def fit(self, data: Iterable[str]) -> None:
        """
        Calls keras.preprocessing.text.Tokenizer.fit_on_texts() on the input 'data'
        """

        self.tokenizer.fit_on_texts(data)

        self.word_index = self.tokenizer.word_index
        self.vocabulary_size = len(self.tokenizer.word_index)

        logging.info(f"Vocabulary size: {self.vocabulary_size}")

    def tokenize(self, data: Iterable[str]) -> np.array:
        """
        Calls:
         keras.preprocessing.text.Tokenizer.texts_to_sequences()
         keras.preprocessing.sequence.pad_sequences()
        sequentially.
        """
        logging.info("Tokenizing sentences")
        return keras_pad_sequences(
            self.tokenizer.texts_to_sequences(data), self.max_sequence_lengh
        )
