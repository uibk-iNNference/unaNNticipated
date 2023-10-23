from posixpath import basename, splitext
from typing import Dict, List
import click
from invoke import run
import os
from os.path import join
import numpy as np

from glob import glob
from innfrastructure import metadata, compare
import json

import pandas as pd

TARGET_MODEL = "cifar10_medium"
TARGET_LAYER = "conv2d_11"
RESULT_DIR = join("results", "parameter_distribution", f"{TARGET_MODEL}-{TARGET_LAYER}")


def load_result(path: str) -> np.ndarray:
    with open(path, "r") as f:
        result_dir = json.load(f)

    raw_predictions = result_dir["prediction"]
    return metadata.convert_json_to_np(raw_predictions)


def get_equivalence_classes_per_prediction(predictions: List[np.array]):
    uniques = []
    equivalence_classes = [
        compare.get_equivalence_class(uniques, prediction) for prediction in predictions
    ]
    return equivalence_classes


def get_host(path: str) -> str:
    name = splitext(basename(path))[0]
    return name


def get_equivalence_classes(current_dir: str) -> List[Dict[str, int]]:
    paths = list(glob(join(current_dir, "*.json")))
    results = [load_result(path) for path in paths]
    assert all([result.shape == results[0].shape for result in results])

    hostnames = [get_host(path) for path in paths]

    model_type, sample_type = basename(current_dir).split("-")

    rows = []
    for i in range(results[0].shape[0]):
        predictions = [r[i].flatten() for r in results]
        eqcs = get_equivalence_classes_per_prediction(predictions)

        rows += [
            {
                "host": host,
                "eqc": eqc,
                "model_type": model_type,
                "sample_type": sample_type,
                "index": i,
            }
            for host, eqc in zip(hostnames, eqcs)
        ]

    return rows


def main():
    rows = []
    for child, _, _ in os.walk(RESULT_DIR):
        if child == RESULT_DIR:
            continue

        rows += get_equivalence_classes(child)

    df = pd.DataFrame(rows)
    groupby = df.groupby("host")["eqc"]
    assert all(groupby.min() == groupby.max())

    print("Equivalence classes are consistent over all model and input types")

    print(groupby.describe())


if __name__ == "__main__":
    main()
