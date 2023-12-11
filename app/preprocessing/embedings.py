import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import tensorflow_hub
import tensorflow_text
from dotenv import load_dotenv
from keras import layers as LA
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

load_dotenv()

import os

EMERGENCY_PATH = Path(os.environ["EMERGENCY_EMBEDINGS_MATRIX_PATH"])


# NOTE: Embedings matrix should have already been paresd and persisted
# I did not know for how long I would stick with this approach so it looks a bit ugly.
# It turns out that I actully followed this approach but did not get time to make this more robust.
def build_embedings_matrix_from_glove(
    embedings_file_path: Path, word_index: Any, max_vocab_size: int
):
    logging.info("Will build embeddings matrix")

    try:
        assert (
            embedings_file_path.exists()
        ), f"Embedings path {embedings_file_path} cannot be resolved"

        file = open(embedings_file_path, "r", encoding="utf8")
    except Exception as err:
        logging.error(
            f"Cannot open emedings file: {embedings_file_path}."
            "Will use cached embedings matrix. Original error {err.args}"
        )

        try:
            with open(EMERGENCY_PATH, "rb") as fl:
                return cloudpickle.load(fl)
        except Exception():
            "Failed to load mergency embedings. This should not have happened"
            raise RuntimeError() from err

    embedded_index = {}

    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embedded_index[word] = vector

    embedded_dim = len(tuple(embedded_index.values())[0])

    embedded_matrix = np.zeros((max_vocab_size, embedded_dim))
    for x, i in word_index.items():
        vector = embedded_index.get(x)
        if vector is not None:
            embedded_matrix[i] = vector

    return embedded_matrix


def embedings_layer_keras(
    input_dim: int, output_dim: int, input_length: int, embeddings_matrix: str
) -> LA.Embedding:
    args = dict(input_dim=input_dim, output_dim=output_dim, input_length=input_length)
    # embedings layer
    if embeddings_matrix is not None:
        args.update(dict(weights=[embeddings_matrix], trainable=False))
    else:
        args["trainable"] = True

    return LA.Embedding(**args)


# NOTE: Belo this point it was lasty minute experimentation
# This would not be dumped like this in a production level code base
class IEncoder(abc.ABC):
    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def tokenize(self, data):
        pass

    @abc.abstractmethod
    def to_keras_layer(self) -> "EncoderLayer":
        pass


class EncoderLayer(LA.Layer):
    def __init__(self, encoder: IEncoder, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return self.tokenize(x)


class TfIdfEncoder(IEncoder):
    def __init__(self, max_features: int):
        self.encoder = TfidfVectorizer(binary=True, max_features=max_features)

    def fit(self, data):
        self.encoder.fit(data)

    def tokenize(self, data):
        return self.encoder.transform(data).toarray()

    def to_keras_layer(self) -> "EncoderLayer":
        return EncoderLayer(self)


import os


@dataclass
class MuseEncoder(IEncoder):
    model_url: str = (
        "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
    )
    cache: dict[str, np.array] = field(default_factory=lambda: {})

    def fit(self, data, batch_size: int = 50, from_cache: bool = True):
        cache_path = os.environ["MUSE_CACHE"]
        if from_cache:
            try:
                embeded = cloudpickle.load(open(cache_path, "rb"))
            except Exception:
                msg = f"Failed to load cached muse embedings from {cache_path}, will compute them"
                logging.warn(msg)

        logging.info("Downloading muse model embeder")
        self.model = tensorflow_hub.load(self.model_url)

        tmp = []
        n_batches = int(data.shape[0] / batch_size) + 1
        for batch in tqdm(range(n_batches)):
            cursor = batch * batch_size
            try:
                queries = data[cursor : cursor + batch_size]
            except IndexError:
                queries = data[cursor:]

            # embeded = self.model(queries)
            print(batch, cursor, cursor + batch_size)
            tmp += [queries]
            # self.cache.update(dict(zip(queries, embeded)))

        import pdb

        pdb.set_trace()

        assert len(data) == len(self.cache), "You did not embeed all your input data"

        if not os.path.exists(cache_path):
            with open(cache_path, "wb") as fl:
                cloudpickle.dump(embeded, fl)

        return embeded

    def tokenize(self, query):
        embeding = self.cache.get(query, None)

        if not embeding:
            embeding = self.model(query)
        else:
            raise RuntimeError("Failed to embed")

        return embeding

    def to_keras_layer(self) -> "EncoderLayer":
        return EncoderLayer(self)
