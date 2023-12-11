import pytest
from app.preprocessing.tokenization import KerasTokenizer
import numpy as np


@pytest.mark.parametrize(
    "data,vocabulary_size,word_index,expected_tokens",
    argvalues=[
        (
            ["ultimate challenge", "ultimate challenge"],
            2,
            {"ultimate": 1, "challenge": 2},
            [(0, 0, 0, 1, 2), (0, 0, 0, 1, 2)],
        ),
        (
            ["entropy always increases"],
            3,
            {"entropy": 1, "always": 2, "increases": 3},
            [(0, 0, 1, 2, 3)],
        ),
        (
            ["this", "is", "a", "corpus"],
            4,
            {"this": 1, "is": 2, "a": 3, "corpus": 4},
            [(0, 0, 0, 0, 1), (0, 0, 0, 0, 2), (0, 0, 0, 0, 3), (0, 0, 0, 0, 4)],
        ),
    ],
    ids=[
        "2 words 2 sents",
        "3 words 1 sent ",
        "4 words 4 sents",
    ],
)
def test_keras_tokenizer(data, vocabulary_size, word_index, expected_tokens):
    tkn = KerasTokenizer(max_sequence_lengh=5)

    tkn.fit(data)

    assert tkn.vocabulary_size == vocabulary_size, "vocabulary sizes dont match"

    assert tkn.word_index == word_index

    for actual, expected in zip(tkn.tokenize(data), expected_tokens):
        assert all(np.array(actual) == np.array(expected)), "tokens dont match"
