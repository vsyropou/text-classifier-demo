"""
Here we define a simple dataclass to hold manage configurations
"""

import os
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from app.constants import (
    INPUT_LAYER_SPECS,
    LOSS_SPECS,
    OPTIMIZER_SPECS,
    OUTPUT_LAYER_SPECS,
    RAW_FEATURES,
    SEED,
    TARGET_FEATURE,
)

load_dotenv()


# TODO: This configurations class deserves a bit more tidying up / refactoring
class WorkflowConf(BaseModel):
    model_config = ConfigDict(frozen=True)
    # TODO:  frozen does not work as I remmeber, plus arguments can be mutated some times
    # investigate how to make ths config class more light weight

    # contorl the seed for reproducbility
    seed: float = Field(default=SEED, frozen=True)

    # feature names
    feature_names: list[str] = Field(default=deepcopy(RAW_FEATURES), frozen=True)
    target_feature: list[str] = Field(default=deepcopy(TARGET_FEATURE), frozen=True)

    # load data specific
    train_data_path: str = Field(
        default=Path(os.environ["DATA_PATH"]) / Path("train.tsv"), frozen=True
    )
    test_data_path: str = Field(
        default=Path(os.environ["DATA_PATH"]) / Path("test.tsv"), forzen=True
    )
    train_test_split: float = Field(default=0.7, frozen=True)

    load_data_csv_seperator: str = Field(default="\t", frozen=True)

    # tuning specific
    tune_model: bool = Field(default=False, frozen=True)

    # preprocessing specific
    max_sequence_length: int = Field(default=10, frozen=True)

    tokenizer_class: str = Field(default="KerasTokenizer", frozen=True)

    stopwords_languages: list[str] = Field(default=["english"], frozen=True)

    # embedings specific
    embedings_file_path: str = Field(
        default=Path(os.environ["STORAGE_PATH"]) / Path("embedings/glove.6B.50d.txt"),
        frozen=True,
    )

    # training specific
    # TODO: Use keras model .to_json to serialize and parameterize the whole model
    training_module: str = Field(default="app.models.cnn", frozen=True)
    builder_class: str = Field(default="KerasConvolutionalBuilder", frozen=True)
    builder_input_specs: list[dict[str, str | int | float]] = Field(
        default=deepcopy(INPUT_LAYER_SPECS)
    )
    builder_output_specs: list[dict[str, str | int | float]] = Field(
        default=deepcopy(OUTPUT_LAYER_SPECS)
    )

    builder_optimizer_specs: dict[str, str | float | int] = Field(
        default=deepcopy(OPTIMIZER_SPECS), frozen=True
    )

    builder_loss_specs: dict[str, str | float | int] = Field(
        default=deepcopy(LOSS_SPECS), frozen=True
    )
    final_activation: str = "softmax"
    train_embedings: bool = False

    epochs: int

    batch_size: int = 2**7

    # experimental / risky switches
    use_weights: bool = False
    use_tensorboard: bool = False
    use_tf_idf: bool = False
    use_muse: bool = False
