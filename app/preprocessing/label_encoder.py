"""
Here we keep the code for encoding our target labels
"""

import abc
import logging
from pprint import pprint
from typing import Iterable

import numpy as np
from pydantic import BaseModel


class ILabelEncoder(abc.ABC):
    @abc.abstractclassmethod
    def fit(self, data: Iterable[int]) -> None:
        pass

    @abc.abstractclassmethod
    def transform(self, data: Iterable[int]) -> np.array:
        pass


class PrimeLabelEncoder(ILabelEncoder, BaseModel):
    mapping: dict[str, int] = None
    max_dim: int = None
    prime_categories: set[str] = None

    """
    Given a string based target feature the following operations are supported:
     - splits the categories by a seperator to sort out combined categories
     - identifies the unique categories
     - multi hot encodes the initial labels

     Example:

        data = ["flight+ground", "flight"]
        lb = LabelEncoder()

        lb.fit(data, seperator="+")
        lb.prime_categories
        2

        lb.mapping
        {"flight": 0, "ground": 1}
        lb.max_dim
        2

        lb.transform(data)
        [
        [1,1],
        [1,0]
        ]
    """

    # @property
    # def mapping(self) -> dict[str, int]:
    #     return self.mapping

    # @property
    # def max_dim(self) -> str:
    #     return self.max_dim

    # @property
    # def max_prime_categories(self) -> int:
    #     return self.prime_categories

    def fit(self, data: Iterable[int], seperator: str = "+") -> None:
        combined_categories = list(map(lambda x: x.split(seperator), set(data)))
        prime_categories = set([prime for c in combined_categories for prime in c])

        logging.info(f"These are the {len(prime_categories)} prime categories:")
        pprint(prime_categories)

        mapping = dict(zip(prime_categories, range(len(prime_categories))))

        self.prime_categories = prime_categories
        self.mapping = mapping
        self.max_dim = len(prime_categories)

    def transform(self, data: Iterable[int]) -> np.array:
        encoded = list(map(lambda x: [self.mapping[i] for i in x.split("+")], data))

        multi_hot_encoded = np.zeros((len(encoded), self.max_dim))

        multi_hot_encoded = [
            [int(int(idx) in enc) for idx in range(self.max_dim)] for enc in encoded
        ]

        return np.array(multi_hot_encoded)
