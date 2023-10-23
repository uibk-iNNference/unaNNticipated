import itertools
import json
import logging
from glob import glob
from os.path import join
from typing import List

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from innfrastructure import compare, metadata, models
from joblib.parallel import Parallel, parallel_backend, delayed


def load_predictions(path: str):
    with open(path, "r") as file:
        result_dict = json.load(file)

    raw_predictions = result_dict["prediction"]
    ret = [metadata.convert_json_to_np(raw) for raw in raw_predictions]
    return ret


def count_equivalence_classes(layer_predictions: List[np.array]):
    flattened = [p.flatten() for p in layer_predictions]
    stacked = np.vstack(flattened)
    unique = np.unique(stacked, axis=0)
    return unique.shape[0]


def get_precision(layer_predictions: List[np.array]):
    # total precision (precision left between all)
    # is minimum pairwise precision
    combinations = itertools.combinations(layer_predictions, 2)
    bit_distances = [compare.remaining_precision(p1, p2) for p1, p2 in combinations]
    precisions = np.subtract(23, bit_distances)
    return np.min(precisions)


def get_l2_distance(layer_predictions: List[np.array], normalize: bool):
    combinations = itertools.combinations(layer_predictions, 2)
    l2_distances = [compare.l2_distance(p1, p2) for p1, p2 in combinations]

    if normalize:
        max_prediction = max([prediction.max() for prediction in layer_predictions])
        min_prediction = min([prediction.min() for prediction in layer_predictions])
        l2_distances = l2_distances / (max_prediction - min_prediction)

    return max(l2_distances)


def get_l0_distance(layer_predictions: List[np.array]):
    combinations = itertools.combinations(layer_predictions, 2)
    l0_distances = [compare.l0_distance(p1, p2) for p1, p2 in combinations]

    return max(l0_distances)

def load_data(model_type: str, sample_index: int, hosts:List[str] = None):
    # load model
    model = models.get_model(model_type.replace("linear","sigmoid")[: -len("_instrumental")])
    # get layer names
    names = [layer.name for layer in model.layers]

    # load results
    glob_expression = join(
        "results",
        "instrumental",
        f"{model_type}-i{sample_index}",
        "*.json",
    )
    result_paths = glob(glob_expression)

    if hosts is not None and len(hosts) > 0:
        path_copy = result_paths
        result_paths = []

        for path in path_copy:
            if any([host in path for host in hosts]):
                result_paths.append(path)

    if len(result_paths) == 0:
        print("No results found")
        return

    # extract predictions
    predictions = [load_predictions(path) for path in result_paths]
    by_layer = list(zip(*predictions))

    # calculate metrics
    def analyze_layer(layer):
        return (
            count_equivalence_classes(layer),
            get_precision(layer),
            get_l2_distance(layer, normalize=True),
            get_l0_distance(layer),
        )

    with parallel_backend("loky"):
        results = Parallel()(delayed(analyze_layer)(layer) for layer in by_layer)
        eqc_counts, precisions, l2_distances, l0_distances = zip(*results)

    # build dataframe
    df = pd.DataFrame(
        {
            "name": names,
            "equivalence_classes": eqc_counts,
            "precision": precisions,
            "l2_distance": l2_distances,
            "l0_distance": l0_distances,
        }
    )

    return names, df

@click.group()
def main():
    pass

@main.command()
@click.argument("model_type")
@click.option("--sample-index", type=int, default=0)
@click.option(
    "--hosts",
    type=str,
    help="Comma separated list of hosts to include.\nOther hosts will be removed",
    default="",
)
def plot(model_type: str, sample_index: int, hosts):
    if not model_type.endswith("_instrumental"):
        logging.warning("Appending '_instrumental' to model name")
        model_type += "_instrumental"

    hosts: List[str] = hosts.split(",")

    names, df = load_data(model_type, sample_index, hosts)
    previous_precision = 23
    precision_reducing_names = []
    precision_increasing_names = []
    previous_eqcs = 1
    eqcs_reducing_names = []
    eqcs_increasing_names = []
    for i, (name, eqcs, precision) in enumerate(zip(names, df.equivalence_classes, df.precision)):
        if precision < previous_precision:
            precision_reducing_names.append(name)

        if precision > previous_precision:
            precision_increasing_names.append(name)

        previous_precision = precision

        if eqcs < previous_eqcs:
            eqcs_reducing_names.append(name)

        if eqcs > previous_eqcs:
            eqcs_increasing_names.append(name)
            print(f"Layer {i} {name} increased EQCs")

        previous_eqcs = eqcs

    reducing_layer_types = set(map(lambda n: '_'.join(n.split("_")[:-1]), precision_reducing_names))
    increasing_layer_types = set(map(lambda n: '_'.join(n.split("_")[:-1]), precision_increasing_names))
    print(f"\nLayer types reducing precision: {reducing_layer_types}")
    print(f"\nLayer types increasing precision: {increasing_layer_types}")

    reducing_layer_types = set(map(lambda n: '_'.join(n.split("_")[:-1]), eqcs_reducing_names))
    increasing_layer_types = set(map(lambda n: '_'.join(n.split("_")[:-1]), eqcs_increasing_names))
    print(f"\nLayer types reducing eqcs: {reducing_layer_types}")
    print(f"\nLayer types increasing eqcs: {increasing_layer_types}")

    # plot
    fig, axes = plt.subplots(3, 1)
    xs = list(range(len(names)))

    labels = [
        "Equivalence classes",
        "Max normalized L0 distance",
        "Max normalized L2 distance",
        "Minimum Precision",
        "Equivalence Classes",
    ]

    if len(hosts) > 0:
        labels = [
            "",
            "Normalized L0 distance",
            "Normalized L2 distance",
            "Remaining precision",
            "Equivalence Classes",
        ]

    if len(hosts) == 0:
        axes[0].plot(xs, df.equivalence_classes, label=labels[0])
    axes[0].plot(xs, df.l0_distance, label=labels[1])
    axes[1].plot(xs, df.l2_distance, label=labels[2])
    axes[2].plot(xs, df.precision, label=labels[3])
    axes[2].plot(xs, df.equivalence_classes, label=labels[4])

    for ax in axes:
        ax.set_xticks(xs)
        ax.legend()
        ax.set_xticklabels(names, rotation=-90)

    suptitle = model_type
    if len(hosts) > 0:
        suptitle += f", filtered to hosts {hosts}"
    fig.suptitle(suptitle)

    # print l2 information for generating images
    print(df.l2_distance.describe())

    plt.show()

@main.command()
@click.argument("model_type")
@click.option("--sample-index", type=int, default=0)
@click.option(
    "--hosts",
    type=str,
    help="Comma separated list of hosts to include.\nOther hosts will be removed",
    default="",
)
def output(model_type: str, sample_index: int, hosts):
    if not model_type.endswith("_instrumental"):
        logging.warning("Appending '_instrumental' to model name")
        model_type += "_instrumental"

    hosts: List[str] = hosts.split(',')
    names, df = load_data(model_type, sample_index, hosts)
    df = df.reset_index()
    print(df.to_csv(header=['index','name','eqcs','precision','l2_dist','l0_dist'],index=False,sep=' '))

if __name__ == "__main__":
    main()
