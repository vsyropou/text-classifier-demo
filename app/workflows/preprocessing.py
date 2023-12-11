"""
Preprocessing workflows.
Workflows group together fucntion calls and handles the artifacts persisstense
"""

from functools import partial
from typing import Optional

from app.artifactory import LocalArtifactory
from app.conf import WorkflowConf
from app.preprocessing.label_encoder import PrimeLabelEncoder
from app.preprocessing.load_data import read_and_reshuffle_data
from app.preprocessing.preprocess_data import inverse_propensity_weights, order_labels
from app.preprocessing.preprocess_data import preprocessor as data_preprocessor


def workflow_preposesing(
    cnf: WorkflowConf,
    artifactory: LocalArtifactory,
    tags: Optional[dict] = None,
):
    """
    Passes the necessary configuration to the trianing related functions
    Handles reading and persisting of artifacts
    """

    # NOTE: Intents were not equally repesented in both sets reshuffling

    train_data, test_data = read_and_reshuffle_data(
        train_data_path=cnf.train_data_path,
        test_data_path=cnf.train_data_path,
        seperator=cnf.load_data_csv_seperator,
        feature_names=cnf.feature_names,
        target_feature=cnf.target_feature,
        train_test_split=cnf.train_test_split,
        seed=cnf.seed,
    )

    x_train = train_data[cnf.feature_names].values[:, 0]
    y_train = order_labels(train_data[cnf.target_feature].values)

    weights = inverse_propensity_weights(y_train)

    x_test = test_data[cnf.feature_names].values[:, 0]
    y_test = order_labels(test_data[cnf.target_feature].values)

    preprocessor = partial(
        data_preprocessor,
        stopwords_languages=cnf.stopwords_languages,
    )

    label_encoder = PrimeLabelEncoder()
    label_encoder.fit(y_train)

    labels_train = label_encoder.transform(y_train)
    labels_test = label_encoder.transform(y_test)

    x_train_processed = preprocessor(data=x_train)
    x_test_processed = preprocessor(data=x_test)

    artifactory.save(preprocessor, "preprocessor")
    artifactory.save(weights, "weights")
    artifactory.save(x_train_processed, "x_train_processed")
    artifactory.save(x_test_processed, "x_test_processed")
    artifactory.save(labels_train, "labels_train")
    artifactory.save(labels_test, "labels_test")
    artifactory.save(label_encoder, "label_encoder")
