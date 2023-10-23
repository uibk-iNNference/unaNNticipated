import json
from os import path
from glob import glob
import click
from innfrastructure import metadata, compare
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib.parallel import Parallel, parallel_backend, delayed


def analyze_directory(directory: str):
    print(f"Analyzing {directory}")

    # parse parameters
    name = path.basename(directory)
    _, fs, k, input_size = [s[1:] for s in name.split("_")]
    i, j, c = input_size.split("x")
    f1, f2 = fs.split("x")
    assert i == j
    assert f1 == c

    f1, f2, c, i, k = [int(v) for v in [f1, f2, c, i, k]]

    # extract predictions
    paths = glob(path.join(directory, "*.json"))
    dicts = [json.load(open(path, "r")) for path in paths]
    predictions = [
        metadata.convert_json_to_np(dict["prediction"])
        for dict in dicts
        # if dict["hostname"] in ["intel-sandy-bridge", "amd-rome"]
    ]

    combinations = list(itertools.combinations(predictions, 2))
    bit_distances = [compare.remaining_precision(p1, p2) for p1, p2 in combinations]
    value_distances = [compare.l0_distance(p1, p2) for p1, p2 in combinations]

    stack = np.vstack([p.flatten() for p in predictions])
    uniques = np.unique(stack, axis=0)
    equivalence_classes = uniques.shape[0]

    # bit_distance = np.mean(bit_distances)
    precisions = np.subtract(23, bit_distances)
    # l0_distance = np.mean(value_distances)

    return [
        {
            "f1": f1,
            "f2": f2,
            "k": k,
            "i": i,
            "multiplications": k * k * i * i * f1 * (f1 + f2),
            "kernel_parameters": f1 * k * k * (c + f2),
            "precision": precision,
            "l0_distance": value_distance,
            "equivalence_classes": equivalence_classes,
            "resolution": equivalence_classes / len(predictions),
        }
        for precision, value_distance in zip(precisions, value_distances)
    ]


def analyze_data(result_dir):
    directories = glob(path.join(result_dir, "*"), recursive=True)
    with parallel_backend("loky"):
        blocks = Parallel(n_jobs=4)(
            delayed(analyze_directory)(directory) for directory in directories
        )
    rows = [row for block in blocks for row in block]
    df = pd.DataFrame(rows)
    df.to_feather(path.join(result_dir, "data.feather"))
    return df


@click.command()
@click.argument(
    "result_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
def main(result_dir: str):
    data_path = path.join(result_dir, "data.feather")
    try:
        df = pd.read_feather(data_path)
    except FileNotFoundError:
        print(f"No existing data found in {data_path} found, generating...")
        df = analyze_data(result_dir)

    # filter for a subset of fs
    # df = df.query("f1 > 1")

    grid = sns.relplot(
        data=df,
        y="precision",
        x="multiplications",
        kind="line",
        hue="f1",
        col="k",
        style="f2",
        markers=True,
        height=8,
        estimator="mean",
        ci=None,
        palette=sns.color_palette("tab10", 3),
    )
    ax = grid.axes[0][0]
    ax.set_xscale("log")
    plt.show()


if __name__ == "__main__":
    main()
