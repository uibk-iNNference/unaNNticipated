from os.path import join, basename
from glob import glob
import json
from innfrastructure import compare, metadata
import invoke
import click

RESULT_DIR = join("results", "max_resolution")


@click.command()
@click.option("--plot/--no-plot", default=False)
@click.option("--use-all/--no-use-all", default=False)
def main(plot: bool, use_all: bool):
    models = [
        "extracted_layer",
        "without_bias",
        "default_distribution",
        "fitted_distribution",
        "f256_k3_i113_c3",
    ]

    if use_all:
        models = [basename(path) for path in glob(join(RESULT_DIR, "*"))]

    # bit of a hack, but it works
    for model in models:
        paths = glob(join(RESULT_DIR, model, "*.json"))

        predictions = []
        for path in paths:
            with open(path, "r") as f:
                result = json.load(f)
                predictions.append(metadata.convert_json_to_np(result["prediction"]))

        unique_predictions = []
        equivalence_classes = [
            compare.get_equivalence_class(unique_predictions, prediction)
            for prediction in predictions
        ]
        print(f"{model}: {max(equivalence_classes)+1} equivalence classes")

        if plot:
            invoke.run(
                f"python main/eval/dendrogram.py '' --base-dir {join(RESULT_DIR, model)} -m bits",
                disown=True,
            )


if __name__ == "__main__":
    main()
