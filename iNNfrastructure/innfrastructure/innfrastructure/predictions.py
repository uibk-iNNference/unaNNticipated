from typing import Union

import numpy as np
import tensorflow.keras.losses as losses
from tensorflow import GradientTape, constant
import tensorflow as tf
from os.path import basename

from . import models, metadata


def get_device_placement(model: tf.keras.Model) -> str:
    # get output tensor # HACK
    full_device_string = model.outputs[-1]._keras_history.layer.weights[0].device
    return f"/{basename(full_device_string)}"


def _disseminate_batch(samples, model):
    rets = []

    for sample in samples:
        current_results = model(np.expand_dims(sample, 0))
        rets.append(current_results)

    if type(rets[0]) is list:
        # switch list shape
        transposed = zip(*rets)
        stacked = list(map(np.vstack, transposed))

        return stacked

    return np.vstack(rets)


def predict(
    model_type: str,
    samples: np.ndarray,
    disseminate_batches: bool = True,
):

    model = models.get_model(model_type)

    if disseminate_batches:
        return _disseminate_batch(samples, model), get_device_placement(model)

    results = model(samples)
    if type(results) is list:
        results = [result.numpy() for result in results]
    else:
        results = results.numpy()

    return results, get_device_placement(model)
