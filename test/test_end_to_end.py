"""
End to end workflow testing. This take a bit longer to run. In a production
setting this would be a slow test that does not run in the premerge tests but every
say a few hours from master branch.

Also you would like to make all the possible configuration combinations and pass them
to the @parameterize decorator above. Since this is for demostration purposes only I
will only made two cases.
"""

from numbers import Number
from typing import Generator

from app.conf import WorkflowConf
from app.workflows.model_building.keras_cnn import workflow_keras_glove_builder
from app.workflows.preprocessing import workflow_preposesing
from app.workflows.training import workflow_training


def test_end_to_end_workflow(benchmark, local_artifactory: Generator):
    conf = WorkflowConf(epochs=1)

    with local_artifactory as artifactory:
        wf_args = dict(cnf=conf, artifactory=artifactory)
        workflow_preposesing(**wf_args)
        workflow_keras_glove_builder(**wf_args)
        workflow_training(**wf_args)

        # load serving model
        model = artifactory.load("serving_model")

        # NOTE: Normally an end to end test should stop here
        # and the servig object should be tested seperatelly with mocking its depndencies.

        # Hoever due to lack of time I include all the tests here.
        # Imagine the follwing tests structured in a single function each and with
        # the all the dependencies of the serving model mocked

        # check that model is not None
        assert model, "Workflow did not produce a model artifact"

        # check has predict method
        assert hasattr(model, "predict"), "Trainded model does not have a predict method"

        # do one prediction
        intent = model.predict("ultimate intent")

        # benchmark the invocation latency
        benchmark(lambda: model.predict("ultimate intent"))

        # In the following we check the response schema
        assert intent, "model responds with None"

        assert type(intent) is list, f"model return type is {type(intent)} should be list"

        assert intent[0], f"model did not make predictions: {intent[0]}"

        assert (
            type(intent[0]) is dict
        ), f"model prediction should be 'dict' got {type(intent[0])} instead"

        assert "label" in intent[0].keys(), "model prediction does not have 'label' field"

        assert (
            "confidence" in intent[0].keys()
        ), "model prediction does not have 'confidence' field"

        assert type(intent[0]["label"]) is str, "model prediction 'label' is not a string"

        assert isinstance(
            intent[0]["confidence"], Number
        ), "model prediction 'confidence' is not a number"
