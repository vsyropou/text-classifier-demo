"""Here we keep functions that we use to load and split data"""

# TODO: iterator typehint
# TODO: validate types
from pathlib import Path
from typing import Iterator

import pandas as pd


def get_csv_data(
    data_path: Path,
    seperator: Path,
    feature_names: list[str],
    target_feature: str,
) -> pd.DataFrame:
    """
    Reads csv file and sperates the features from the target columns
    """

    data = pd.read_csv(
        data_path,
        sep=seperator,
        names=[*feature_names, target_feature],
    )

    x = data[feature_names].values[:, 0]
    y = data[target_feature].values

    return x, y


def read_and_reshuffle_data(
    train_data_path: Path,
    test_data_path: Path,
    seperator: Path,
    feature_names: list[str],
    target_feature: str,
    train_test_split: float,
    seed: int = 678,
) -> pd.DataFrame:
    """
    Reads both train and test files, concatenates them together and does a stratified
    sampling based on the target column. The split ratio is specified by 'train_test_split'.
    This is to ensure that the labes during training and testing overlap
    """

    data = pd.concat(
        [
            pd.read_csv(
                p,
                sep=seperator,
                names=[*feature_names, target_feature],
            )
            for p in [train_data_path, test_data_path]
        ]
    ).reset_index()

    train_data = data.groupby(target_feature).sample(
        frac=train_test_split, random_state=seed
    )

    test_data = data.loc[data.index.difference(train_data.index)]

    return train_data.sample(frac=1, random_state=seed), test_data.sample(
        frac=1, random_state=seed
    )
