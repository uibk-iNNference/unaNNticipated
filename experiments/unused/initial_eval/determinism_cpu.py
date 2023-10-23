import os
import matplotlib.pyplot as plt
import seaborn as sns
from innfrastructure import metadata
from sklearn.decomposition import PCA
import streamlit as st
from glob import glob
import pandas as pd
import numpy as np
import glob
import json

MODEL_TYPES = [
    "cifar10_small",
    "cifar10_medium",
    "cifar10_large",
    "minimal",
]


def plot_histograms():

    df = build_histogram()

    g = sns.catplot(
        data=df,
        col="hostname",
        row="model_type",
        x="equivalence_class",
        y="count",
        hue="sample_index",
        kind="bar",
        aspect=3,
    )
    g.fig.subplots_adjust(top=0.9)
    st.pyplot(g)

    target_dir = os.path.join("results", "eval", "determinism")
    os.makedirs(target_dir, exist_ok=True)
    plt.savefig(os.path.join(target_dir, f"cpus.png"))


def build_histogram():
    results = [load_converted_results(path) for path in get_paths()]

    df = pd.DataFrame(
        columns=["hostname", "model_type" "equivalence_class", "count", "sample_index"]
    )

    rows = []
    for result in results:
        predictions = result["distribution_predictions"]
        hostname = result["hostname"]
        model_type = result["model_type"]
        sample_index = result["distribution_sample_index"]

        _, counts = np.unique(predictions, return_counts=True, axis=0)
        counts = np.sort(counts)[::-1]

        for eq_class, count in enumerate(counts):
            rows.append(
                {
                    "hostname": hostname,
                    "model_type": model_type,
                    "equivalence_class": eq_class,
                    "count": count,
                    "sample_index": sample_index,
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["model_type", "hostname"])


def load_converted_results(path: str):
    with open(path) as f:
        result_dict = json.load(f)
        to_convert = ["prediction", "distribution_predictions"]

        for key in to_convert:
            values = metadata.convert_json_to_np(result_dict[key])
            if len(values.shape) > 2:
                values = values.reshape((values.shape[0], -1))
            result_dict[key] = values

        # TODO: clean cpu name
        return result_dict


def get_paths():
    return glob.glob(f"results/main/*cpu*.json")


def plot_pca():
    df = generate_pca()

    g = sns.relplot(
        data=df,
        x="x",
        y="y",
        col="model_type",
        hue="hostname",
        size="count",
        style="sample_index",
    )
    g.fig.subplots_adjust(top=0.9)
    st.pyplot(g)


def generate_pca():
    # load all result files

    results = [load_converted_results(path) for path in get_paths()]

    dfs = []

    for model_type in MODEL_TYPES:
        filtered_results = list(
            filter(lambda r: r["model_type"] == model_type, results)
        )

        pca_data = np.vstack(
            [result["distribution_predictions"] for result in filtered_results]
        )

        # normalize over ALL data
        mean = np.mean(pca_data, axis=0)
        mean_diff = pca_data - mean
        mean_min = mean_diff.min(axis=0)
        mean_max = mean_diff.max(axis=0)

        def normalize(data):
            diff = data - mean
            denominator = mean_max - mean_min
            denominator[denominator == 0] = 1.0
            return (diff - mean_min) / denominator

        normalized_pca_data = normalize(pca_data)

        pca = PCA(n_components=2, random_state=42)
        pca.fit(normalized_pca_data)

        # transform predictions per model_type
        for result in filtered_results:
            predictions = result["distribution_predictions"]
            normalized_predictions = normalize(predictions)

            transformed_predictions = pca.transform(normalized_predictions)
            uniques, counts = np.unique(
                transformed_predictions, return_counts=True, axis=0
            )
            xs = uniques[:, 0]
            ys = uniques[:, 1]

            current_df = pd.DataFrame(
                {
                    "x": xs,
                    "y": ys,
                    "count": counts,
                    "model_type": result["model_type"],
                    "hostname": result["hostname"],
                    "sample_index": result["distribution_sample_index"],
                }
            )
            dfs.append(current_df)

    full_df = pd.concat(dfs)
    return full_df.sort_values(["model_type", "hostname"])


def visualize():
    st.markdown("## CPU Determinism")

    st.markdown(
        """
        ### Do CPUs produce deterministic results?
        """
    )

    plot_histograms()

    st.markdown(
        """
        Looks like they do!
        
        Let's check how diverse our CPUs are."""
    )
    plot_pca()


if __name__ == "__main__":
    visualize()
