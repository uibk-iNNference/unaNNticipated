from innfrastructure.predictions import predict
from innfrastructure import models
import numpy as np


def test_prediction(model):
    models.models["test_model"] = model
    sample = np.random.random((1,) + model.input_shape[1:])

    predicted, device = predict("test_model", sample)

    expected = model(sample).numpy()

    assert np.all(predicted == expected)


def test_full_batch_prediction(model):
    models.models["test_model"] = model
    sample = np.random.random((5,) + model.input_shape[1:])

    predicted, _ = predict("test_model", sample, disseminate_batches=False)
    expected = model(sample).numpy()

    assert np.all(predicted == expected)


def test_disseminated_batch_prediction(model):
    models.models["test_model"] = model
    samples = np.random.random((5,) + model.input_shape[1:])

    predicted, _ = predict("test_model", samples, disseminate_batches=True)
    expected_list = [model(np.expand_dims(s, 0)) for s in samples]
    expected = np.vstack(expected_list)

    assert np.all(predicted == expected)
