import itertools
import json
import logging
from glob import glob
from os.path import join, basename

import click
from innfrastructure import compare, metadata, models
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def load_predictions(path: str):
    with open(path, "r") as file:
        result_dir = json.load(file)

    raw_predictions = result_dir["prediction"]
    ret = [metadata.convert_json_to_np(raw) for raw in raw_predictions]
    return ret


@click.command()
@click.argument("model_type", type=str)
def main(model_type: str):
    if not model_type.endswith("_instrumental"):
        logging.warning("Appending '_instrumental' to model name")
        model_type += "_instrumental"

    # load model
    model = models.get_model(model_type)
    # get layer names
    names = [layer.name for layer in model.layers]

    # load results
    glob_expression = join("results", "main", "gcloud", f"*{model_type}*.json")
    result_paths = glob(glob_expression)
    hostnames = [basename(path).split("-")[0] for path in result_paths]

    # extract predictions
    predictions = [load_predictions(path) for path in result_paths]
    by_layer = list(zip(*predictions))

    rows = []
    for i, (layer_name, layer_predictions) in enumerate(zip(names, by_layer)):
        unique_predictions = []
        equivalence_classes = [
            compare.get_equivalence_class(unique_predictions, prediction)
            for prediction in layer_predictions
        ]

        for hostname, eqc in zip(hostnames, equivalence_classes):
            rows.append(
                {
                    "layer_index": i,
                    "hostname": hostname,
                    "layer": layer_name,
                    "equivalence_class": eqc,
                }
            )

    df = pd.DataFrame(rows)

    sns.relplot(
        data=df, x="layer_index", hue="hostname", y="equivalence_class", kind="line"
    )
    plt.show()


if __name__ == "__main__":
    main()
