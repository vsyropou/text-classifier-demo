"""
This is the main file of our training procedure

Example usage:

Run from the top repo directory:
'python app/train_model.py' 

"""
import logging
import os
import shutil
from datetime import datetime
from pprint import pprint

from dotenv import load_dotenv

from app.artifactory import LocalArtifactory
from app.cli import parse_cli_args
from app.conf import WorkflowConf
from app.workflows.model_building.keras_cnn import workflow_keras_glove_builder
from app.workflows.model_building.keras_muse import workflow_keras_muse_builder
from app.workflows.model_building.keras_tfidf import workflow_keras_tfidf_builder
from app.workflows.preprocessing import workflow_preposesing
from app.workflows.training import workflow_training

load_dotenv()
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)

if __name__ == "__main__":
    cli = parse_cli_args()

    # initialize artifact storage
    timestamp = str(datetime.now())  # this is used to keep all retrained models
    artifactory = LocalArtifactory(uri=cli.artifactory_uri, extra=timestamp)

    # initialize workflow conf and tags
    conf = WorkflowConf(
        tune_model=cli.tune or False,
        epochs=cli.epochs,
        use_tensorboard=cli.use_tensorboard,
    )
    tags = {"timestamp": timestamp, "user_tags": cli.tags}

    # trigger training
    logging.info(f"These are the model tags: {tags}")
    logging.info(f"This is the pipeline configuration:")
    logging.info("Executing workfow with args:")
    pprint(conf.model_dump())

    # tensorboard
    if cli.use_tensorboard:
        # TODO: this is gettign too long puting in a function
        logging.info("Cleaning tensorboard sink")
        tb_path = os.environ["TENSORBOARD_SINK"]
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
            os.makedirs(tb_path)

    # compile workflow
    wf_args = dict(cnf=conf, artifactory=artifactory, tags=tags)

    workflow_preposesing(**wf_args)
    if conf.use_tf_idf:
        workflow_keras_tfidf_builder(**wf_args)
    elif conf.use_muse:
        workflow_keras_muse_builder(**wf_args)
    else:
        workflow_keras_glove_builder(**wf_args)
    workflow_training(**wf_args)

    artifactory.save(conf.model_dump_json(), "conf")
    artifactory.save(tags, "tags")

    # if you are retraining or this is first run then set principal model
    if cli.replace_principal or len(os.listdir(cli.artifactory_uri)) == 1:
        artifactory.move(destination="principal-model")
