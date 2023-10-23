from glob import glob
from os import path as pth
import click
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import json
from scipy.cluster import hierarchy

from innfrastructure import metadata, compare

RESULT_DIR = pth.join("results", "max_resolution")
CONFIGURATIONS = [pth.basename(dir) for dir in glob(pth.join(RESULT_DIR, "*"))]


@click.command()
@click.argument("configuration", type=click.Choice(choices=CONFIGURATIONS))
def main(configuration: str):
    paths = glob(pth.join(RESULT_DIR, configuration, "*.json"))
    results = []
    for path in paths:
        with open(path, "r") as f:
            results.append(json.load(f))

    predictions = {
        result["hostname"]: metadata.convert_json_to_np(result["prediction"])
        for result in results
    }

    key_pairs = list(itertools.combinations(predictions.keys(), 2))
    distances = [
        compare.remaining_precision(predictions[k1], predictions[k2])
        for k1, k2 in key_pairs
    ]

    Z = hierarchy.linkage(distances)

    hierarchy.dendrogram(Z, labels=list(predictions.keys()), orientation="left")
    plt.xlabel("Rounded mantissa bits")
    plt.title(configuration)
    plt.subplots_adjust(right=0.75)
    plt.xlim(24)
    plt.xticks([23, 20, 15, 10, 5, 0])

    plt.show()


if __name__ == "__main__":
    main()
