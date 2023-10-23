import random
import pytest
from os.path import join
from innfrastructure import config

from innfrastructure import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


@pytest.fixture
def model():
    # create and save a model
    model = Sequential([
        Dense(random.randrange(100)),
        Dense(random.randrange(100))
    ])
    model.build(input_shape=(None, 10))
    return model


def assert_model_equality(actual: Sequential, expected: Sequential):
    for l1, l2 in zip(actual.layers, expected.layers):
        assert l1.get_config() == l2.get_config()


def test_model_loading(tmp_path, model):
    target_path = join(tmp_path, "loading.h5")
    model.save(target_path)
    config.MODEL_DIR = tmp_path

    # actual test
    loaded = models.get_model("loading")
    assert_model_equality(loaded, model)


def test_model_reloading(tmp_path, model):
    target_path = join(tmp_path, "reloading.h5")
    model.save(target_path)

    config.MODEL_DIR = tmp_path


    # first load
    loaded = models.get_model("reloading")
    assert_model_equality(loaded, model)

    # second load
    reloaded = models.get_model("reloading")
    assert_model_equality(reloaded, model)
