import pytest
import os
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


@pytest.fixture
def dirname(request):
    """Directory name test fixture.
    As we're going to need the containing directory for many of our tests,
    I wrote this small fixture. It gets the file name from the builtin
    request fixture, and computes the dirname of it. The basedir fixture can
    be requested by giving the test function a `dirname` parameter.

    Args:
        request (pytest.fixture): The pytest builtin request fixture

    Returns:
        str: the directory name (full absolute path) to the containing
            directory of the current test
    """
    return os.path.dirname(os.path.dirname(request.fspath))


@pytest.fixture
def model():
    """
    create a test model
    """
    model = Sequential([Dense(random.randrange(100)), Dense(random.randrange(100))])
    model.build(input_shape=(None, 10))
    return model
