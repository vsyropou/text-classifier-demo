"""Here we define some cli arguments that our main workflow apps can use.

Examples:

    'python app/train_model.py'
        trains a random forest model w/o replacing the principal model
    'python app/model.py --replace-principal --tune'
        trains and tunes the model and updates the model

"""
import argparse
import os

from dotenv import load_dotenv

load_dotenv()


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="FluxCapasitor", description="Correlates Quantum States")

    # define cli arguments
    parser.add_argument("--tune", action="store_true", help="turn tuning on/off")
    parser.add_argument("--use-tensorboard", action="store_true", help="use tensorboard")
    parser.add_argument(
        "--replace-principal",
        action="store_true",
        help="replaces principal model with the model from this run",
    )
    parser.add_argument(
        "--artifactory-uri",
        default=os.environ.get("ARTIFACTORY_URI", None),
        help="specify a filde to save artifacts",
    )
    parser.add_argument("-t", "--tags", action="append", required=False)
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument(
        "--pool-size",
        type=int,
        default=4,
        required=False,
        help="only available for running experiments",
    )

    # parse cmd args
    cli = parser.parse_args()
    print(f"Received commands: {cli}")
    assert (
        cli.artifactory_uri
    ), "Please specify 'artifactory-uri' or set 'ARTIFACTORY_URI' env variable"

    return cli
