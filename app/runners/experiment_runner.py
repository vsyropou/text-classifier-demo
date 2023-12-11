"""
This is the main file for running experiments

Example usage:

Run from the top repo directory:
    python app/experiment_runner.py  --epochs 20 --pool-size <number of threads>

"""
import hashlib
import logging
import os
from datetime import datetime
from multiprocessing.pool import ThreadPool
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv

from app.artifactory import LocalArtifactory
from app.cli import parse_cli_args
from app.conf import WorkflowConf
from app.workflows.model_building.keras_cnn import workflow_keras_glove_builder
from app.workflows.model_building.keras_tfidf import workflow_keras_tfidf_builder
from app.workflows.preprocessing import workflow_preposesing
from app.workflows.training import workflow_training

load_dotenv()


FORMAT = "[%(asctime)s %(filename)s:%(lineno)s]%(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def run_experiment(epxeriment_conf: dict[str, str | WorkflowConf]):
    ts = epxeriment_conf["ts"]
    tags = epxeriment_conf["tags"]
    conf = epxeriment_conf["conf"]
    conf_json = conf.model_dump_json().encode("utf-8")

    # initialize artifact storage
    experiment_hash = hashlib.md5(conf_json).hexdigest()

    experiments_uri = Path(os.environ["STORAGE_PATH"]) / Path("experiments")
    extra_uri = Path(ts) / Path(experiment_hash)

    artifactory = LocalArtifactory(uri=experiments_uri, extra=extra_uri)

    tags.update({"timestamp": str(datetime.now())})

    # trigger training
    logging.info(f"These are the model tags: {tags}")
    logging.info(f"This is the experiment hash: {experiment_hash}")
    logging.info(f"This is the pipeline configuration:")
    logging.info("Executing workfow with args:")
    pprint(conf.model_dump())

    wf_args = dict(cnf=conf, artifactory=artifactory, tags=tags)

    workflow_preposesing(**wf_args)
    if conf.use_tf_idf:
        workflow_keras_tfidf_builder(**wf_args)
    else:
        workflow_keras_glove_builder(**wf_args)
    workflow_training(**wf_args)

    artifactory.save(conf.model_dump_json(), "conf")
    artifactory.save(tags, "tags")


if __name__ == "__main__":
    cli = parse_cli_args()

    # we group same executions of configurations (experiments) by timestamp
    timestamp = str(datetime.now())

    experiments = [
        {
            "tags": {"name": "default"},
            "conf": WorkflowConf(epochs=cli.epochs),
            "ts": timestamp,
        },
        {
            "tags": {"name": "ULTIMATE"},
            "conf": WorkflowConf(epochs=cli.epochs, use_weights=True),
            "ts": timestamp,
        },
    ]

    with ThreadPool(cli.pool_size) as pool:
        results = pool.map(run_experiment, experiments)
