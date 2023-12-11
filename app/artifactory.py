""" Here we define the code that saves and loads throughout the training and serving process"""
import abc
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import cloudpickle


class IArtifactory(abc.ABC):
    """
    Support artifactory interface for multiple backends
    """

    @abc.abstractclassmethod
    def save(self):
        pass

    @abc.abstractclassmethod
    def load(self):
        pass

    @abc.abstractclassmethod
    def move(self):
        pass


class LocalArtifactory(IArtifactory, dict):
    """
    A simple way to save and load artifacts locally using cloudpickle
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.uri})"

    def __init__(self, uri: str, extra: str = None):
        setattr(self, "uri", uri)
        setattr(self, "extra", extra)

        if not Path(self.uri).exists():
            logging.warn(
                f"Provided artifactory uri '{self.uri}' cannot be resolved, creating path"
            )
            os.makedirs(self.uri)

        # We append extra partioning, e.g. timestamp, to the path in order to
        # differenciate between runs
        if self.extra:
            self.uri = f"{self.uri}/{self.extra}"

        if not os.path.exists(self.uri):
            os.makedirs(self.uri)

        self.uri = Path(self.uri)

    def save(self, obj: Any, path: str) -> None:
        """
        Saves an object in artifactory uri under the filename 'path'
        """
        logging.info(f"Saving artifact in {self.uri}/{path}")

        # "cache" in memory for this session
        self[path] = obj

        # pickle artifact
        with open(self.uri / path, "wb") as fl:
            cloudpickle.dump(obj, fl)

    def load(self, path: str) -> Any:
        logging.info(f"Searching for artifact: '{path}'")
        if self.get(path, None) is not None:  # retrieve from memory if it exists
            logging.info(f"Found cached artifact in: '{path}'")
            artifact = self[path]
        else:  # unpickle
            logging.info(
                f"Retrieving artifact with tags: '{path}' from storage location: '{self.uri}'"
            )

            with open(self.uri / path, "rb") as fl:
                artifact = cloudpickle.load(fl)

        return artifact

    def move(self, destination: str) -> None:
        # delete destination path and recreate it
        other_path = self.uri.parent / Path(destination)
        if os.path.exists(other_path):
            shutil.rmtree(other_path)

        # copy current path to other path
        this_path = self.uri

        logging.info(f"Replacing: '{this_path}' with '{other_path}'")
        shutil.copytree(this_path, other_path)

    def save_callable(self, callable: Callable):
        # the callable owns the persistance method this arfiactory owns the path
        # we use this for saving matplotlib figures
        try:
            callable(self.uri)
        except Exception:
            raise IOError(f"Cannot save external file to '{self.uri}'")
        logging.info("Saved callable")

    def list_artifacts(self) -> list[str]:
        return os.listdir(self.uri)
