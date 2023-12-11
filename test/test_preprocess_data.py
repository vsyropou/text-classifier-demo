import pytest
import numpy as np

from app.preprocessing.preprocess_data import remove_digits, order_labels, inverse_propensity_weights

digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.mark.parametrize("digit", argvalues=digits, ids=digits)
def test_remove_remove_digits(digit):
    sentence = [
        "This 1 is a 2 sentence 4 that 3 has a 5 digits 6 statered 7 arround 8 9 0"
    ]

    clean_sentence = remove_digits(sentence)

    assert digit not in clean_sentence, f"digit {digit} was not removed"


@pytest.mark.parametrize(
    "raw,excpected",
    argvalues=[
        ("zzz+aaa", "aaa+zzz"),
        ("aaa+zzz", "aaa+zzz"),
        ("zz+aa+bb", "aa+bb+zz"),
        ("aa+bb+zz", "aa+bb+zz"),
    ],
    ids=["2 wrong", "2 correct", "3 wrong", "3 correct"],
)
def test_order_labels(raw, excpected):
    result = order_labels(np.array([raw]))[0]

    assert (
        result == excpected
    ), f"Labels, {result}, are wrongly ordered, should be {excpected} "


@pytest.mark.parametrize(
    "raw,excpected",
    argvalues=[
        (["a", "a", "b"], [0.5, 0.5, 1.0]),
        (["a", "b"], [1, 1]),
        (["a", "a", "a"], [1 / 3.0, 1 / 3.0, 1 / 3.0]),
        (["a", "b", "c"], [1, 1, 1]),
    ],
    ids=["3 diff", "2 same", "3 same", "3 diff"],
)
def test_inverse_propensity_weights(raw, excpected):
    result = inverse_propensity_weights(np.array(raw))

    assert np.allclose(
        result, excpected, atol=0.0001
    ), f"weights, {result} are wrong. should be {np.array(excpected)}"
