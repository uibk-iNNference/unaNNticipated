from innfrastructure import compare, models, metadata
import pandas as pd
import numpy as np


def test_comparison():
    prediction_a = np.random.random((3, 10))
    prediction_b = np.random.random((3, 10))
    device_names = ["prediction_a1", "prediction_a2", "prediction_b"]
    equivalence_classes = ['A','A','B']

    row = {"Experiment": "Experiment"}
    for name, eq_class in zip(device_names, equivalence_classes):
        row[name] = eq_class
    expected = pd.DataFrame([row])

    layout = {
        "Experiment": dict(
            zip(device_names, [prediction_a, prediction_a, prediction_b])
        )
    }
    compared = compare.generate_comparison(layout)

    pd.testing.assert_frame_equal(compared, expected)
