import functools
import itertools
import json
from glob import glob
from itertools import combinations
import math
from os import path

import numpy as np
import click

from innfrastructure import compare, metadata, models


def split_basename(target_path):
    return path.basename(target_path).split("-")


def extract_all_layer_outputs(model_path):
    json_paths = glob(path.join(model_path, "*.json"))
    results = [json.load(open(p)) for p in json_paths]

    all_layer_outputs = list(
        map(
            lambda result: [
                metadata.convert_json_to_np(layer_result)
                for layer_result in result["prediction"]
            ],
            results,
        )
    )
    assert all(
        [
            len(all_layer_outputs[0]) == len(layer_outputs)
            for layer_outputs in all_layer_outputs
        ]
    )

    return all_layer_outputs


# %% count which activation layers increase RP, which leave it constant, and which decrease
def count_activation_influences(model_path, all_layer_outputs, activation_layers):
    model_name, _ = split_basename(model_path)
    remaining_precisions = []
    for i in range(len(all_layer_outputs[0])):
        current_layer_outputs = [
            layer_outputs[i] for layer_outputs in all_layer_outputs
        ]
        assert all(type(v) == np.ndarray for v in current_layer_outputs)

        # generate all pairwise remaining precisions
        output_combinations = list(combinations(current_layer_outputs, 2))
        pairwise_remaining_precisions = [
            23 - compare.remaining_precision(v1, v2) for v1, v2 in output_combinations
        ]

        min_remaining_precision = min(pairwise_remaining_precisions)
        remaining_precisions.append(min_remaining_precision)

    increasing, constant, decreasing = 0, 0, 0

    for i in activation_layers.keys():
        assert i > 0

        if remaining_precisions[i] > remaining_precisions[i - 1]:
            increasing += 1
        elif remaining_precisions[i] == remaining_precisions[i - 1]:
            constant += 1
        else:
            decreasing += 1

    activation = "sigmoid" if "sigmoid" in model_name else "relu"
    return activation, increasing, constant, decreasing


def get_activation_layers(model_path):
    # find actually existing variation of the model
    model_name, sample_index = split_basename(model_path)

    m = models.get_model(model_name.split("_instrumental")[0])

    activations = {}
    for i, layer in enumerate(m.layers):
        try:
            activations[i] = layer.activation.__name__
        except AttributeError:
            # might be an activation layer itself
            class_name = layer.__class__.__name__
            if class_name in ["ReLU", "Sigmoid"]:
                activations[i] = class_name.lower()

    # filter for sigmoid and relu
    filtered_activations = dict(
        filter(lambda elem: elem[1] in ["sigmoid", "relu"], activations.items())
    )
    return filtered_activations


@click.group()
def main():
    pass


@main.command()
def precision():
    model_names = ["cifar10_medium", "cifar10_small"]
    latex_model_names = ["\\mmedium", "\\msmall"]
    for model_name, latex_model_name in zip(model_names, latex_model_names):
        print_row(model_name, latex_model_name)


def print_row(model_name, latex_model_name=None):
    if latex_model_name is None:
        latex_model_name = model_name

    result_dir = path.join("results", "instrumental")

    act, *relu_stats = get_activation_influences(
        f"{model_name}_instrumental", result_dir
    )
    assert act == "relu"
    act, *sig_stats = get_activation_influences(
        f"{model_name}_sigmoid_instrumental", result_dir
    )
    assert act == "sigmoid"
    print(f"{latex_model_name} & ", end="")
    counts = itertools.chain(relu_stats, sig_stats)

    print(" & ".join(map(lambda x: f"{x:.3f}", counts)), end="")
    print("\\\\")


def get_activation_influences(model_name, result_dir):
    sample_dirs = glob(path.join(result_dir, model_name + "*"))
    activation_layers = get_activation_layers(sample_dirs[0])

    activation = None
    increasing, constant, decreasing = 0, 0, 0
    for sample_dir in sample_dirs:
        all_layer_outputs = extract_all_layer_outputs(sample_dir)
        act, inc, cons, dec = count_activation_influences(
            sample_dir, all_layer_outputs, activation_layers
        )

        if activation is None:
            activation = act
        else:
            assert activation == act

        increasing += inc
        constant += cons
        decreasing += dec

    # average results
    increasing /= len(sample_dirs)
    constant /= len(sample_dirs)
    decreasing /= len(sample_dirs)
    return activation, increasing, constant, decreasing


@main.command()
def deviations():
    model_names = ["cifar10_medium"] #, "cifar10_small"]
    latex_model_names = ["\\mmedium", "\\msmall"]

    result_dir = path.join("results", "instrumental")
    for i, (model_name, latex_model_name) in enumerate(
        zip(model_names, latex_model_names)
    ):
        if i > 0:
            print("\\\\")

        print(f"{latex_model_name} \\\\")

        variants = ["", "_sigmoid"]
        (relu_stats, activation_layers), (sigmoid_stats, _) = [
            get_model_deviation_stats(
                result_dir, model_name + variant + "_instrumental"
            )
            for variant in variants
        ]

        columns = []
        for i, (idx, activation) in enumerate(activation_layers.items()):
            column = [
                idx,
                f"{relu_stats[idx-1]:.1f}",
                f"{relu_stats[idx]:.1f}",
                f"{sigmoid_stats[idx-1]:.1f}",
                f"{sigmoid_stats[idx]:.1f}",
            ]
            # print(
                # f"\\qquad {idx} & {relu_stats[idx-1]:.1f}\\,\\% & {relu_stats[idx]:.1f}\\,\\% & {sigmoid_stats[idx-1]:.1f}\\,\\% & {sigmoid_stats[idx]:.1f}\\,\\% \\\\"
            # )

        for row in zip(*columns):
            print(" & ".join(map(str, row)), end="")
            print("\\\\")

    print("\\bottomrule") # again, include bottomrule in generated .tex


def get_model_deviation_stats(result_dir, extended_model_name):
    sample_dirs = glob(path.join(result_dir, extended_model_name + "*"))

    sample_stats = [
        calculate_deviations_per_sample(sample_dir) for sample_dir in sample_dirs
    ]

    # average results
    avg_stats = np.mean(sample_stats, axis=0) * 100
    activation_layers = get_activation_layers(sample_dirs[0])
    return avg_stats, activation_layers


def calculate_deviations_per_sample(sample_dir):
    all_layer_outputs = extract_all_layer_outputs(sample_dir)

    layer_stats = []
    for layer_outputs in zip(*all_layer_outputs):
        deviations, possible = count_deviations(layer_outputs)
        layer_stats.append(deviations / possible)

    return layer_stats


def count_deviations(layer_outputs):
    assert all(type(v) == np.ndarray for v in layer_outputs)
    individual_deviations = [o != layer_outputs[0] for o in layer_outputs]
    deviations = functools.reduce(np.logical_or, individual_deviations)
    return np.sum(deviations), np.prod(deviations.shape)


if __name__ == "__main__":
    main()
