from typing import Dict, List
import pandas as pd
import numpy as np


def l0_distance(v1: np.array, v2: np.array):
    assert v1.dtype == v2.dtype
    assert len(v1[0]) == len(v2[0])

    dev = np.sum(v1 != v2) / np.prod(v1.shape)
    return dev


def l2_distance(v1: np.array, v2: np.array):
    assert v1.dtype == v2.dtype
    assert len(v1[0]) == len(v2[0])

    return np.linalg.norm(v1 - v2)


# WARNING: I realized that this function is backwards. It should return mantissa_size - i.
# some of the existing scripts rely on this error and fix it, so I'm leaving it for now
def remaining_precision(v1: np.array, v2: np.array) -> float:
    assert v1.dtype == v2.dtype

    mantissa_size = None
    bits1 = None
    bits2 = None
    if v1.dtype == np.float32:
        mantissa_size = 23
        bits1 = v1.view(np.int32)
        bits2 = v2.view(np.int32)
    elif v1.dtype == np.float16:
        mantissa_size = 10
        bits1 = v1.view(np.int16)
        bits2 = v2.view(np.int16)
    elif v1.dtype == np.float64:
        mantissa_size = 52
        bits1 = v1.view(np.int64)
        bits2 = v2.view(np.int64)

    assert mantissa_size is not None

    i = None
    for i in range(mantissa_size + 1):
        if np.all(bits1 >> i == bits2 >> i):
            break
    assert i is not None

    return i


def get_equivalence_class(
        unique_predictions: List, prediction
) -> int:
    """Get equivalence class for a prediction as a running index.
    Updates the list of unique predictions as needed.

    Args:
        unique_predictions (List[np.ndarray]): unique predictions seen before
        prediction (np.ndarray): the new prediction

    Returns:
        int: the equivalence class of the new prediction
    """
    candidate_class = 0
    # test if any of the existing classes matches the prediction
    for unique_prediction in unique_predictions:
        if np.all(prediction == unique_prediction):
            break
        candidate_class += 1

    # if none do, append the prediction
    if candidate_class == len(unique_predictions):
        unique_predictions.append(prediction)

    return candidate_class


def generate_comparison(layout: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    def map_to_letter(equivalence_class):
        return chr(65 + equivalence_class)

    """take config in form of:
    [
     "cifar10":
      {
          "server": prediction,
          "consumer": prediction,
      }
    ]
    use "cifar10" as row identifier, "server" as column identifier,
        and the prediction as basis for equivalence classes. Generate comparison table

    Args:
        layout (Dict[str, Dict[str, str]]): the comparison layout to use.

    Returns:
        pd.DataFrame: the resulting comparison dataframe
    """
    rows = []

    for row_name, values in layout.items():
        row = {"Experiment": row_name}
        unique_predictions = []

        for column_name, prediction in values.items():
            equivalence_class = get_equivalence_class(unique_predictions, prediction)

            row[column_name] = map_to_letter(equivalence_class)

        rows.append(row)

    return pd.DataFrame(rows)
