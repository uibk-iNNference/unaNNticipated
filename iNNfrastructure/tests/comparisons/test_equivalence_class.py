from innfrastructure import compare
import pandas as pd
import numpy as np


def test_equivalence_class():
    prediction_a = np.random.random((3, 10))
    prediction_b = np.random.random((3, 10))
    expected = [0, 0, 1]

    compared = []
    unique_predictios = []
    for prediction in [prediction_a, prediction_a, prediction_b]:
        compared.append(compare.get_equivalence_class(unique_predictios, prediction))
    assert expected == compared
