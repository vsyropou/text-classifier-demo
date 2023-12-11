"""
Keras cnn with tfidf embedings workflow.
Workflows group together fucntion calls and handles the artifacts persisstense
"""

#NOTE: This is work inprogress, features might not work properly

import importlib
from typing import Optional

from app.artifactory import LocalArtifactory
from app.conf import WorkflowConf
from app.preprocessing.embedings import TfIdfEncoder
from app.models.cnn import KerasConvolutionalBuilder


def workflow_keras_tfidf_builder(
    cnf: WorkflowConf,
    artifactory: LocalArtifactory,
    tags: Optional[dict] = None,
):
    x_train_processed = artifactory.load("x_train_processed")
    x_test_processed = artifactory.load("x_test_processed")
    labels_train = artifactory.load("labels_train")

    embeder = TfIdfEncoder(max_features=10)  # just trying something fast here
    embeder.fit(x_train_processed)

    x_train_tokenized = embeder.tokenize(x_train_processed)
    x_test_tokenized = embeder.tokenize(x_test_processed)

    # Lack of time. TODO: trainer functions loads the correct model
    models_module = importlib.import_module(cnf.training_module)

    model_builder: KerasConvolutionalBuilder = getattr(
        models_module, cnf.builder_class
    )(
        input_dim=10,
        input_length=cnf.max_sequence_length,
        n_target_classes=len(labels_train[0]),
        final_activation=cnf.final_activation,
    )
    model = model_builder.build(
        input_layers_specs=cnf.builder_input_specs,
        outpput_layer_specs=cnf.builder_output_specs,
        optimizer_specs=cnf.builder_optimizer_specs,
        loss_specs=cnf.builder_loss_specs,
    )

    artifactory.save(model, "model")
    artifactory.save(embeder, "tokenizer")
    artifactory.save(x_train_tokenized, "x_train_tokenized")
    artifactory.save(x_test_tokenized, "x_test_tokenized")
