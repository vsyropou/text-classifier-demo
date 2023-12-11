from typing import Optional

from app.artifactory import LocalArtifactory
from app.conf import WorkflowConf
from app.models.serving import ServingModel
from app.models.trainer import train_model


def workflow_training(
    cnf: WorkflowConf,
    artifactory: LocalArtifactory,
    tags: Optional[dict] = None,
):
    model = artifactory.load("model")
    weights = artifactory.load("weights")
    preprocessor = artifactory.load("preprocessor")
    labels_train = artifactory.load("labels_train")
    labels_test = artifactory.load("labels_test")
    label_encoder = artifactory.load("label_encoder")
    x_train_tokenized = artifactory.load("x_train_tokenized")
    x_test_tokenized = artifactory.load("x_test_tokenized")
    tokenizer = artifactory.load("tokenizer")

    test_metrics, train_metrics, metric_names = train_model(
        model=model,
        x_train=x_train_tokenized,
        y_train=labels_train,
        fit_args=dict(
            epochs=cnf.epochs,
            batch_size=cnf.batch_size,
            validation_data=(x_test_tokenized, labels_test),
            sample_weight=weights if cnf.use_weights else None,
            verbose=2,
        ),
        use_tensorboard=cnf.use_tensorboard,
    )

    serving_model = ServingModel(
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        model=model,
    )

    artifactory.save(serving_model, "serving_model")
    artifactory.save(test_metrics, "test_metrics")
    artifactory.save(train_metrics, "train_metrics")
    artifactory.save(metric_names, "metric_names")
