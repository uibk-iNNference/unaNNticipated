from glob import glob
from os.path import join, dirname
from innfrastructure import metadata, compare
import json
from invoke import config

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


RESULT_DIR = join("results", "dendrogram_consistency")


def analyze_data():
    paths = glob(join(RESULT_DIR, "**", "*.json"), recursive=True)

    rows = []
    uniques = {}
    for path in paths:
        with open(path, "r") as f:
            result = json.load(f)

            path_parts = path.split("/")
            configuration = path_parts[-3]

            conf_parts = configuration.split("_")
            f, k = (int(part[1:]) for part in conf_parts[1:3])
            i = int(conf_parts[-1].split("x")[1])

            run_info = path_parts[-2]
            r, s = [int(part[1:]) for part in run_info.split("_")]

            try:
                current_uniques = uniques[dirname(path)]
            except KeyError:
                current_uniques = []
                uniques[dirname(path)] = current_uniques

            prediction = metadata.convert_json_to_np(result["prediction"])
            equivalence_class = compare.get_equivalence_class(
                current_uniques, prediction
            )

            rows.append(
                {
                    "f": f,
                    "k": k,
                    "i": i,
                    "r": r,
                    "s": s,
                    "equivalence_class": equivalence_class,
                    "configuration": configuration,
                    "hostname": result["hostname"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_feather(join(RESULT_DIR, "data.feather"))
    return df


def main(f: int, k: int, i: int):

    try:
        df = pd.read_feather(join(RESULT_DIR, "data.feather"))
    except FileNotFoundError:
        df = analyze_data()

    # df = df.query(f"f=={f} and k=={k} and i=={i}")

    sns.relplot(
        data=df,
        y="equivalence_class",
        x="s",
        hue="hostname",
        style="r",
        # row="r",
        col="configuration",
        kind="line",
    )

    plt.show()


if __name__ == "__main__":
    main(3, 6, 125)
