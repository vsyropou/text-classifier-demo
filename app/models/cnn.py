"""
    Flexible builder class of keras sequential models for text based networks.
"""

import logging

import numpy as np
from keras import Sequential
from keras import layers as LA
from keras import losses as LS
from keras import metrics as MT
from keras import optimizers as OP
from pydantic import BaseModel

from app.models.base import IModelBuilder, Model


# NOTE: We could also extract parametrizable model templates by using
# the native serialization of keras. Having a formated string template behind a function
class KerasConvolutionalBuilder(IModelBuilder, BaseModel):
    """
    Keras model building wrapper for text based models
    Inputs:
        input_dim: The number of input nodes
        input_length: Sequence length
        n_target_classes: The dimension of the output fully connected layer
        final_activation: The activation function of the output layer
    """

    input_dim: int
    input_length: int
    n_target_classes: int
    final_activation: str

    def build(
        self,
        input_layers_specs: list[dict[str, str | int | float]],
        outpput_layer_specs: list[dict[str, str | int | float]],
        optimizer_specs: dict[str, str | int | float],
        loss_specs: dict[str, str | int | float],
        weights: np.array = None,
        embedings_layer: LA.Embedding | LA.Lambda = None,
    ) -> Model:
        """
            Parses the model specs into a sequential keras model and returns it.
            The method is essentially a wrapper arround the standard flow keras model building.
            The input and output layers are seperated by a Flatten keras layer.

            Inputs:

                input_layers_specs: A list of the specs of each keras layer. The specs format maps 1-1 with the kers layer signatures
                outpput_layer_specs: Same as input but for the output
                optimizer_specs: arguments provied to the keras optimizer class
                loss_specs: arguments provided to the keras loss class
                weights: weights to be used for weighting the loss and metrics during training
                embedings_layer: keras embedings layer, if not provided an input layer is used instead

            Example:
                builder_instance.build(
                input_layers_specs=[
                    dict(klass="Conv1D", filters=2**3, kernel_size=2**2, activation="selu"),
                    dict(klass="MaxPooling1D", pool_size=2)
                    ]
                outpput_layer_specs=[
                    dict(units=16, activation="selu")
                    ],
                optimizer_specs={"name": "Adam", "learning_rate": 0.0001},
                loss_specs={"name": "CategoricalCrossentropy"}
            )
        For the embeding layer you can use the following helper function:
            from app.embedings import embedings_layer_keras
        """
        # convolutional layers
        convolutional_layers = [
            getattr(LA, spec.pop("klass"))(**spec) for spec in input_layers_specs
        ]

        # linear layers
        linear_layers = []
        if outpput_layer_specs:
            linear_layers += [LA.Dense(**spec) for spec in outpput_layer_specs]

        # sequence layers
        layers = []
        if embedings_layer:
            layers += [embedings_layer]
        else:
            layers += [LA.Input(shape=(self.input_dim, self.input_length))]

        layers += [
            *convolutional_layers,
            LA.Flatten(),
            *linear_layers,
            # TODO: Maybe some DropOut as well
            LA.Dense(self.n_target_classes, activation=self.final_activation),
        ]

        model = Sequential(layers=layers)

        # loss
        loss = getattr(LS, loss_specs.get("name"))(**loss_specs)

        # optimizer
        optimizer = getattr(OP, optimizer_specs.get("name"))(**optimizer_specs)

        # compile
        metrics = [MT.Precision(), MT.Recall(), MT.AUC()]
        args = dict(
            optimizer=optimizer,
            loss=loss,
        )

        if weights:
            args["weighted_metrics"] = metrics
            args["loss_weights"] = weights
        else:
            args["metrics"] = metrics

        model.compile(**args)

        logging.info("Here is the model sumamry")

        print(model.summary())

        return Model(base_model=model)
