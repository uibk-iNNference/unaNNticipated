import json
from os.path import join, basename
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
    name = basename(directory)
    _, f, k, input_size = [s[1:] for s in name.split("_")]
    i, j, c = input_size.split("x")
    assert i == j
    f, c, i, k = [int(v) for v in [f, c, i, k]]

    # extract predictions
    paths = glob(join(directory, "*.json"))
    dicts = []
    for path in paths:
        try:
            dicts.append(json.load(open(path, "r")))
        except json.decoder.JSONDecodeError:
            raise ValueError(f"Could not parse json from {path}")

    predictions = [
        metadata.convert_json_to_np(dict["prediction"])
        for dict in dicts
        # if dict["hostname"] in ["intel-sandy-bridge", "amd-rome"]
    ]

    combinations = list(itertools.combinations(predictions, 2))
    bit_distances = [compare.remaining_precision(p1, p2) for p1, p2 in combinations]
    l0_distances = [compare.l0_distance(p1, p2) for p1, p2 in combinations]
    l2_distances = [compare.l2_distance(p1, p2) for p1, p2 in combinations]

    stack = np.vstack([p.flatten() for p in predictions])
    uniques = np.unique(stack, axis=0)
    equivalence_classes = uniques.shape[0]

    # bit_distance = np.mean(bit_distances)
    precisions = np.subtract(23, bit_distances)
    # l0_distance = np.mean(value_distances)

    return [
        {
            "c": c,
            "f": f,
            "k": k,
            "i": i,
            "multiplications": c * f * k * k * i * i,
            "kernel_parameters": c * f * k * k,
            "precision": precision,
            "l0_distance": l0_distance,
            "l2_distance": l2_distance,
            "equivalence_classes": equivalence_classes,
            "resolution": equivalence_classes / len(predictions),
        }
        for precision, l0_distance, l2_distance in zip(
            precisions, l0_distances, l2_distances
        )
    ]


def analyze_data(result_dir):
    directories = glob(join(result_dir, "*"), recursive=True)
    with parallel_backend("loky"):
        blocks = Parallel(n_jobs=3)(
            delayed(analyze_directory)(directory) for directory in directories
        )
    rows = [row for block in blocks for row in block]
    df = pd.DataFrame(rows)
    df.to_feather(join(result_dir, "data.feather"))
    return df


@click.command()
@click.argument(
    "result_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.option(
    "-y",
    type=click.Choice(
        ["precision", "l0_distance", "l2_distance", "equivalence_classes", "resolution"]
    ),
    default="precision",
)
@click.option("--estimator", type=click.Choice(["mean", "min", "max"]), default="mean")
@click.option("--log_y/--no-log_y", type=bool, default=False)
@click.option("--extra-query", type=str, default=None)
def main(result_dir: str, y: str, estimator: str, log_y: bool, extra_query: str = None):
    data_path = join(result_dir, "data.feather")
    try:
        df = pd.read_feather(data_path)
    except FileNotFoundError:
        print(f"No existing data found in {data_path} found, generating...")
        df = analyze_data(result_dir)

    # filter for a subset of fs
    df = df.query("f == 1 or f == 3")
    df = df.query("c == 3")

    if extra_query is not None:
        df = df.query(extra_query)

    grid = sns.relplot(
        data=df,
        y=y,
        x="multiplications",
        kind="line",
        hue="k",
        col="c",
        style="f",
        markers=True,
        height=8,
        estimator=estimator,
        ci=None,
        palette=sns.color_palette("tab10", 6),
    )
    ax = grid.axes[0][0]
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
