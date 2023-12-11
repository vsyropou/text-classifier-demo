"""
Keras cnn with preptrained or optinally pretrained embedings.
Workflows group together fucntion calls and handles the artifacts persisstense
"""

import importlib
from typing import Optional

from app.artifactory import LocalArtifactory
from app.conf import WorkflowConf
from app.models.cnn import KerasConvolutionalBuilder
from app.preprocessing import tokenization
from app.preprocessing.embedings import (
    build_embedings_matrix_from_glove,
    embedings_layer_keras,
)

# TODO: Workflow context gets local variables and perisists them automatically.


def workflow_keras_glove_builder(
    cnf: WorkflowConf,
    artifactory: LocalArtifactory,
    tags: Optional[dict] = None,
):
    x_train_processed = artifactory.load("x_train_processed")
    x_test_processed = artifactory.load("x_test_processed")
    labels_train = artifactory.load("labels_train")

    # TODO: No obeject instansiation or fit emethods inside the workflow
    # TODO: modularize tokenization and label encoding in a function
    tokenizer = getattr(tokenization, cnf.tokenizer_class)(
        max_sequence_lengh=cnf.max_sequence_length
    )
    tokenizer.fit(x_train_processed)

    x_train_tokenized = tokenizer.tokenize(x_train_processed)

    x_test_tokenized = tokenizer.tokenize(x_test_processed)

    embeddings_matrix = build_embedings_matrix_from_glove(
        embedings_file_path=cnf.embedings_file_path,
        word_index=tokenizer.word_index,
        max_vocab_size=len(tokenizer.word_index) + 1,
    )

    embedings_layer = embedings_layer_keras(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=len(embeddings_matrix[0]),
        input_length=cnf.max_sequence_length,
        embeddings_matrix=embeddings_matrix,
    )
    # Lack of time. TODO: trainer functions loads the correct model
    models_module = importlib.import_module(cnf.training_module)

    model_builder: KerasConvolutionalBuilder = getattr(models_module, cnf.builder_class)(
        input_dim=len(tokenizer.word_index) + 1,
        input_length=cnf.max_sequence_length,
        n_target_classes=len(labels_train[0]),
        final_activation=cnf.final_activation,
    )

    model = model_builder.build(
        input_layers_specs=cnf.builder_input_specs,
        outpput_layer_specs=cnf.builder_output_specs,
        optimizer_specs=cnf.builder_optimizer_specs,
        loss_specs=cnf.builder_loss_specs,
        embedings_layer=embedings_layer,
    )

    artifactory.save(model, "model")
    artifactory.save(tokenizer, "tokenizer")
    artifactory.save(embeddings_matrix, "embeddings_matrix")
    artifactory.save(tokenizer.word_index, "word_index")
    artifactory.save(x_train_tokenized, "x_train_tokenized")
    artifactory.save(x_test_tokenized, "x_test_tokenized")
