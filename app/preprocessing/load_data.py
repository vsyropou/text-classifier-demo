"""Here we keep functions that we use to load and split data"""

import logging
import os
from pathlib import Path

import datasets
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def from_huggingface(
    name: str,
    feature_names: list[str],
    target_feature: str,
    train_test_split: float,
    seed: int,
    from_cache: bool = True,
):
    hg_cache = os.environ["HG_DATASETS_CACHE"]

    path = Path(hg_cache) / Path(name)

    if from_cache:
        logging.info(f"Attempting to load {name} dataset hg cache")
        if not os.path.exists(path):
            logging.warn(f"Could not find '{name}' in cache, calling the datasets api")
            dataset = datasets.load_dataset(name)
            dataset.save_to_disk(path)
        else:
            dataset = datasets.load_from_disk(path)

    return reshuffle_data(
        data=pd.concat(
            [
                dataset["train"].to_pandas()[feature_names],
                dataset["test"].to_pandas()[target_feature],
            ]
        ).reset_index(),
        target_feature=target_feature,
        train_test_split=train_test_split,
        seed=seed,
    )


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


def reshuffle_data(
    data: pd.DataFrame, target_feature: str, train_test_split: float, seed=int
) -> pd.DataFrame:
    train_data = data.groupby(target_feature).sample(frac=train_test_split, random_state=seed)

    test_data = data.loc[data.index.difference(train_data.index)]

    return train_data.sample(frac=1, random_state=seed), test_data.sample(
        frac=1, random_state=seed
    )


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
    Reads both train and test files, concatenates them together and does a
    stratified sampling based on the target column. The split ratio is
    specified by 'train_test_split'. This is to ensure that the labes during
    training and testing overlap
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

    return reshuffle_data(
        data=data, target_feature=target_feature, train_test_split=train_test_split, seed=seed
    )
