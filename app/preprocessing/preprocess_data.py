"""
Here we remove stop words and digits and other check class imbalance
"""
import logging
import re
from pprint import pprint
from string import punctuation
from typing import Iterable

import nltk
import numpy as np
from nltk.corpus import stopwords


def remove_stopwords(data: Iterable[str], stopwords=set[str]) -> list[str]:
    return list(
        map(
            lambda x: " ".join([wrd for wrd in x.split() if wrd not in stopwords]), data
        )
    )


def remove_digits(data: Iterable[str]) -> list[str]:
    return list(map(lambda x: re.sub("[0-9]", "", x), data))


def remove_punktuation(data: Iterable[str]) -> list[str]:
    return [w.lower() for w in data if w.lower() and w.lower() not in punctuation]


def preprocessor(
    data: Iterable,
    stopwords_languages: list[str],
) -> np.array:
    """
    Just a combiner function that calls 'remove_stopwords' and 'remove_digits' functions
    """
    for language in stopwords_languages:
        try:
            words = stopwords.words(language)
        except LookupError as err:
            logging.warn("stopwords module from nltk seems to be missing, installing")
            nltk.download("stopwords")
            words = stopwords.words(language)

        logging.info(f"Removing {language} stop words")
        data = remove_stopwords(data=data, stopwords=words)

    logging.info("removing punktuation")

    logging.info("droping digits")
    data = remove_digits(data=data)

    return np.array(data)


def order_labels(data: Iterable[int], seperator: str = "+") -> np.array:
    """
    Splits labels, orders them alhpabetically and then re joins them
    """
    ordered = np.array([f"{seperator}".join(sorted(x.split(seperator))) for x in data])
    logging.info(f"Orddering combined categories")
    return ordered


def inverse_propensity_weights(data: np.array) -> np.array:
    """
    Computes the frequency of each word in return a mapping of the inverse of the frequencies
    Example:
        data = ["cat1", "cat1, "cat2"]
        inverse_propensity_weights(data)
        {"cat_1": 0.5, ""cat2": 1.}
    """
    counts = dict(zip(*np.unique(data, return_counts=True)))

    logging.info(f"These are the label counts in the data:")
    pprint(counts)

    mapping = dict(
        map(lambda it: (it[0], 1.0 / it[1] if it[1] != 0 else 0.0), counts.items())
    )

    logging.info(f"Here are the inverse propensity weights:")
    pprint(mapping)

    return np.array(list(map(lambda x: mapping[x], data)))
