#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import models, layers

from innfrastructure.predictions import get_device_placement


def test_device_placement():
    expected_device = "/device:CPU:0"

    with tf.device(expected_device):
        model = models.Sequential([layers.Dense(10, input_shape=(10,))])
        device = get_device_placement(model)
        assert expected_device == device
