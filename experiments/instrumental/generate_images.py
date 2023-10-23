import itertools
import json
import os
from glob import glob
from os.path import basename, join
from typing import List, Tuple

import click
from matplotlib import colors
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from innfrastructure import compare, metadata, models
from tensorflow.keras.preprocessing.image import save_img


def load_predictions(path: str) -> List[np.array]:
    with open(path, "r") as f:
        result = json.load(f)

    predictions = result["prediction"]
    if isinstance(predictions, List):
        return [metadata.convert_json_to_np(p) for p in predictions]

    return [metadata.convert_json_to_np(predictions)]


def process_layer(i, layer, p1, p2, base_dir):
    print(f"Processing layer {layer.name}...")
    if len(p1.shape) == 2:
        # skip everything after last convolution
        return

    output_dir = join(base_dir, f"{i:03}-{layer.name}")
    os.makedirs(output_dir, exist_ok=True)

    diff = p1 - p2

    abs_max = np.max(np.abs(diff))

    norm = colors.Normalize(vmin=-abs_max, vmax=abs_max)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap="inferno")

    for f in range(diff.shape[-1]):
        current_filter = diff[0, :, :, f]

        fig, ax = plt.subplots()
        ax.imshow(current_filter, norm=norm, cmap="inferno")
        fig.colorbar(mappable)

        target_path = join(output_dir, f"filter_{f:03}.png")

        fig.savefig(target_path)
        plt.close()

        if f >= 10:
            break


@click.command()
@click.argument("target_model", type=str)
@click.option("--sample-index", type=int, default=0)
@click.option(
    "--hosts",
    type=str,
    help="Comma separated list of hosts to include.\nOther hosts will be removed",
)
def main(target_model: str, sample_index: int, hosts):
    if not target_model.endswith("_instrumental"):
        print("WARNING: Appending _instrumental to model name")
        target_model += "_instrumental"

    model = models.get_model(target_model)
    config_name = f"{target_model}-i{sample_index}"
    glob_expression = join("results", "instrumental", config_name, "*.json")
    paths = glob(glob_expression)

    hosts: List[str] = hosts.split(",")
    if len(hosts) > 0:
        path_copy = paths
        paths = []

        for path in path_copy:
            if any([host in path for host in hosts]):
                paths.append(path)

    if len(paths) != 2:
        print(f"Can only deal with two paths, found {len(paths)}")
        return -1

    all_predictions = [load_predictions(path) for path in paths]
    by_layer = zip(*all_predictions)

    image_dir_name = "images"
    if len(hosts) > 0:
        image_dir_name += f"_{hosts}"

    base_dir = join("results", "instrumental", config_name, image_dir_name)

    for i, (layer, (p1, p2)) in enumerate(zip(model.layers, by_layer)):
        process_layer(i, layer, p1, p2, base_dir)

    # add input
    inputs = all_predictions[0][0]
    target_path = join(base_dir, "input.png")
    save_img(target_path, inputs[0])


if __name__ == "__main__":
    main()
