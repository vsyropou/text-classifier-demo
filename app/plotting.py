"""
Some very basic learning curve plotting to facilitate presentation in jupyter
"""

from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np

# NOTE: This functions work in progress. Not suited for any production purposes.


def learning_curve(
    train_lc: Iterable[float], test_lc: Iterable[float], name: str
) -> Callable:
    x = np.arange(1, len(train_lc) + 1)

    fig = plt.figure(figsize=(10, 6))
    plt.subplot(1, 1, 1)
    plt.plot(
        x,
        train_lc,
        "b-",
        label=f"Training Set {name}",
    )
    plt.plot(
        x,
        test_lc,
        "r-",
        label=f"Test Set {name}",
    )
    plt.title(f"{name} Learning Curve")
    plt.legend(loc="upper right")
    plt.xlabel("epochs")
    plt.ylabel(name)
    # plt.xticks(x)

    fig.tight_layout()

    fig.show()

    return lambda path: plt.savefig(
        f"{path}/learning_curve_{name}.pdf", format="pdf", bbox_inches="tight"
    )


def learning_curve_subplot(
    ax,
    train_lc: Iterable[float],
    test_lc: Iterable[float],
    metric_name: str,
    train_name: str = "train",
    test_name: str = "test",
) -> None:
    x = np.arange(1, len(train_lc) + 1)

    ax.plot(
        x,
        train_lc,
        "b-",
        label=f"{train_name}",
    )
    ax.plot(
        x,
        test_lc,
        "r-",
        label=f"{test_name}",
    )
    ax.set_title(f"{metric_name} Learning Curve")
    ax.legend(loc="upper right")
    ax.set_xlabel("epochs")
    ax.set_ylabel(metric_name)
    # ax.set_xticks(x)
