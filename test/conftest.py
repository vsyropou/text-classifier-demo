"""
Here we configure our fixtures
"""

import tempfile
from contextlib import contextmanager
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.api.api import app
from app.artifactory import LocalArtifactory


@pytest.fixture(scope="module")
def client() -> Generator:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="function")
@contextmanager
def local_artifactory() -> Generator:
    tmpdir = tempfile.TemporaryDirectory()
    try:
        yield LocalArtifactory(uri=tmpdir.name)
    finally:
        tmpdir.cleanup()
