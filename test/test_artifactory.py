import os
from pathlib import Path
from typing import Generator

import cloudpickle


def test_artifactory_path(local_artifactory: Generator):
    with local_artifactory as artifactory:
        os.path.exists(artifactory.uri), f"path '{artifactory.uri}' does not exist"


def test_artifactory_save(local_artifactory: Generator):
    with local_artifactory as artifactory:
        obj = ["I want to be persisted"]
        where = "here"
        path = artifactory.uri / where

        artifactory.save(obj, where)

        # path exists
        assert os.path.exists(path), f"path '{path}' does not exist"

        # try to retrieve
        with open(path, "rb") as fl:
            try:
                artifact = cloudpickle.load(fl)
            except Exception as err:
                raise AssertionError(
                    f"Failed to retrieve object from '{path}'"
                ) from err

        # match
        assert obj == artifact, "orignal and pickld objects do not match"


def test_artifactory_load(local_artifactory: Generator):
    with local_artifactory as artifactory:
        obj = ["I want to be persisted"]
        where = "here"
        path = artifactory.uri / where

        artifactory.save(obj, where)

        # try to retrieve
        try:
            artifact = artifactory.load(where)
        except Exception as err:
            raise AssertionError(f"Failed to retrieve object from '{path}'") from err

        # match
        assert obj == artifact, "orignal and pickld objects do not match"


def test_artifactory_move(local_artifactory: Generator):
    with local_artifactory as artifactory:
        obj = ["I want to be persisted"]
        where = "here"
        path = artifactory.uri / where

        destination = "move-here"

        # persist
        artifactory.save(obj, where)

        # retreieve
        artifact = artifactory.load(where)

        # move
        try:
            artifactory.move(destination)
        except Exception as err:
            raise AssertionError(
                f"Failed to move object from '{where}' to {destination}"
            ) from err

        # new location exists
        new_path = artifactory.uri.parent / Path(destination) / where

        assert os.path.exists(new_path), f"New location '{new_path}' does not exist"

        # can load from new location
        try:
            moved_artifact = artifactory.load(new_path.name)
        except Exception as err:
            raise AssertionError(f"Failed to load artifact from new location") from err

        # new and old pickeld objects match
        assert artifact == moved_artifact, "orignal and pickld objects do not match"
