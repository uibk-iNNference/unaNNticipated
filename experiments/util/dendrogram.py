import functools
import os

import click
from glob import glob
from os.path import join, basename, splitext
from typing import Callable, List, Dict, Tuple
import json
import numpy as np
from innfrastructure import metadata, compare
import itertools
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from innfrastructure.compare import (
    l0_distance,
    remaining_precision,
    get_equivalence_class,
)
import pandas as pd


def parse_gpu_result(current_dict: Dict):
    predictions = [
        metadata.convert_json_to_np(pred)
        for pred in current_dict["distribution_predictions"]
    ]
    uniques, counts = np.unique(predictions, return_counts=True, axis=0)
    prediction, _ = sorted(zip(uniques, counts), key=lambda x: x[1])[-1]
    return prediction


def parse_cpu_result(current_dict: Dict):
    try:
        return metadata.convert_json_to_np(current_dict["distribution_predictions"][0])
    except KeyError:
        prediction = current_dict["prediction"]
        return metadata.convert_json_to_np(prediction)


def parse_instrumental_result(current_dict: Dict):
    predictions = current_dict["prediction"]
    return metadata.convert_json_to_np(predictions[-1])


def load_paths(paths: List[str]) -> Dict[str, np.array]:
    ret = {}

    for path in paths:
        with open(path, "r") as current_file:
            current_dict = json.load(current_file)

        if "nvidia" in path:
            prediction = parse_gpu_result(current_dict)
        elif "instrumental" in path:
            prediction = parse_instrumental_result(current_dict)
        else:
            prediction = parse_cpu_result(current_dict)

        hostname = current_dict["hostname"]
        is_good = False
        for vendor in ["nvidia", "amd", "intel"]:
            if vendor in hostname.lower():
                is_good = True
                break

        if not is_good:
            hostname = splitext(basename(path))[0]
        ret[hostname] = prediction

    return ret


METRIC_FUNCTIONS = {"values": l0_distance, "bits": remaining_precision}
SORT_ORDER = [
    "rome",
    "milan",
    "broadwell",
    "haswell",
    "ivy",
    "sandy",
    "cascade",
    "sky",
    "ice",
]
Y_SPACING = 5


@click.group()
def main():
    pass


@main.command()
@click.argument("paths", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--distance-metric",
    "-m",
    type=click.Choice(METRIC_FUNCTIONS.keys(), case_sensitive=False),
    default="bits",
)
@click.option("--output-path", "-o", default=None)
def pyplot(paths, distance_metric: str, output_path:str):
    results = load_paths(paths)
    hostnames = list(results.keys())
    distances = [
        METRIC_FUNCTIONS[distance_metric](a, b)
        for a, b in itertools.combinations(results.values(), 2)
    ]
    z = hierarchy.linkage(distances)
    hierarchy.dendrogram(z, labels=hostnames, orientation="left")
    plt.subplots_adjust(right=0.7)
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)


@main.command()
@click.argument("paths", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--distance-metric",
    "-m",
    type=click.Choice(METRIC_FUNCTIONS.keys(), case_sensitive=False),
    default="bits",
)
@click.option("--filter-hosts/--no-filter-hosts", default=False)
@click.option("--labels/--no-labels", default=True)
@click.option("--calculate/--no-calculate", default=False)
def latex(
    paths: List[str],
    distance_metric: str,
    labels: bool,
    filter_hosts: bool,
    calculate: bool,
):
    results = load_paths(paths)

    keys = list(results.keys())
    if calculate:
        distances = [
            compare.remaining_precision(a, b)
            for a, b in itertools.combinations(results.values(), 2)
        ]
        z = hierarchy.linkage(distances)
        dendrogram = hierarchy.dendrogram(
            z, orientation="left", labels=keys, no_plot=True
        )
        sorted_keys = list(reversed(dendrogram["ivl"]))
    else:
        sort_indices = [get_sort_index(key) for key in keys]
        sorted_keys, _ = zip(*sorted(zip(keys, sort_indices), key=lambda pair: pair[1]))

    if labels:
        for i, hostname in enumerate(sorted_keys):
            print(
                f"\\providecommand{{\\hostname{to_tikz(i)}}}{{ {clean_hostname(hostname, filter_hosts)} }}"
            )

    y_max = Y_SPACING * len(sorted_keys)
    y_coords = {k: y_max - i * Y_SPACING for i, k in enumerate(sorted_keys)}

    initial_values = []
    mantissas = []
    for i, (host, y) in enumerate(y_coords.items()):
        v, mantissa = convert(results[host])
        mantissas.append(mantissa)
        initial_values.append(v)

        line = f"\\draw ({mantissa}, {y}) -- ++(2,0)"
        if labels:
            line += f" node[hostname] {{ \\hostname{to_tikz(i)} }}"
        line += ";"
        print(line)
    assert all([m == mantissas[0] for m in mantissas])
    print(f"% mantissa: {mantissas[0]}")
    unique_predictions = []
    eqcs = [
        compare.get_equivalence_class(unique_predictions, p) for p in results.values()
    ]
    print(f"% {max(eqcs) + 1} eqcs")

    coords = pd.DataFrame(
        {
            "x": mantissas,
            "y": y_coords.values(),
            "value": initial_values,
            "host": y_coords.keys(),
        }
    )
    for i in range(mantissa + 1):
        if len(coords) == 1:
            break
        current_values = coords.value.map(lambda v: v >> i << i)
        unique_values = []
        eqcs = pd.Series(
            [compare.get_equivalence_class(unique_values, v) for v in current_values]
        )
        groupby = coords.groupby(eqcs)

        new_x = mantissa - i

        new_coords = []
        for _, sub_df in groupby:
            if len(sub_df) == 1:
                entry = sub_df.iloc[0]
                new_coords.append({"x": entry.x, "y": entry.y, "value": entry.value})
                continue

            ys = sub_df.y
            xs = sub_df.x
            mid_y = ys.min() + (ys.max() - ys.min()) / 2

            new_coords.append({"x": new_x, "y": mid_y, "value": sub_df.iloc[0].value})
            for x, y in zip(xs, ys):
                print(f"\\draw ({x}, {y}) -- ({new_x}, {y});")

            print(f"\\draw ({new_x}, {ys.min()}) -- ({new_x}, {ys.max()});")

        coords = pd.DataFrame(new_coords)

    min_x = mantissa - i

    if labels:
        print("\\begin{scope}[solid,black]")
        print(
            f"\\draw[->] ({min_x}-1, 0) -- ({mantissa}+1, 0) node[midway,below=3mm] {{\\scriptsize Remaining precision}};"
        )
        for i in range(min_x, mantissa):
            if i % 5 == 0:
                print(f"\\draw ({i}, 0) node[below] {{\\tiny {i}}} -- +(0,1mm);")
            else:
                print(f"\\draw[thin] ({i}, 0) -- +(0,0.5mm);")

        print("\\end{scope}")


def filter_path(path: str) -> bool:
    parts = splitext(path)[0].split("-")
    for part in parts:
        try:
            num_cores = int(part)
            return num_cores == 2
        except ValueError:
            continue
    return True


def get_sort_index(name: str):
    for i, candidate in enumerate(SORT_ORDER):
        if candidate in name:
            return i
    raise ValueError("No match found")


def convert(v: np.ndarray):
    mantissa_size = None
    bits = None
    if v.dtype == np.float32:
        mantissa_size = 23
        bits = v.view(np.int32)
    elif v.dtype == np.float16:
        mantissa_size = 10
        bits = v.view(np.int16)
    elif v.dtype == np.float64:
        mantissa_size = 52
        bits = v.view(np.int64)

    assert mantissa_size is not None
    return bits, mantissa_size


def to_tikz(x):
    return chr(ord("a") + x)


def clean_hostname(hostname, filter_hosts):
    parts = hostname.split("-")
    parts = map(lambda part: part[0].upper() + part[1:], parts)
    parts = map(lambda part: part.upper() if part.lower() in ["amd"] else part, parts)
    parts = filter(lambda part: not part.lower().startswith("float"), parts)
    if filter_hosts:
        parts = filter(lambda part: not _is_int(part), parts)
    return " ".join(parts)


def _is_int(s: str):
    try:
        int(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
