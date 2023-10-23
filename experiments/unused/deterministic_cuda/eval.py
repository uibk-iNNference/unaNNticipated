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
from os.path import join

MODEL_TYPES = [
    "cifar10_small",
    "cifar10_medium",
    "cifar10_large",
    "minimal",
]


def load_unique(path: str):
    with open(path, "r") as f:
        result = json.load(f)

    predictions = metadata.convert_json_to_np(result["distribution_predictions"])
    uniques = np.unique(predictions, axis=0)

    gpu_name = result["device"]["physical_description"]["name"]
    gpu_name = metadata.clean_gpu_name(gpu_name)

    row = [
        len(uniques),
        result["model_type"],
        gpu_name,
        result["hostname"],
        result["distribution_sample_index"],
    ]
    result_df = pd.DataFrame(
        [row], columns=["count", "model_type", "gpu", "hostname", "sample_index"]
    )

    return result_df


def load_uniques():
    paths = get_paths()

    dfs = [load_unique(path) for path in paths]

    full_df = pd.concat(dfs)
    return full_df.sort_values(["model_type", "gpu", "hostname", "sample_index"])


def plot_uniques():
    df = load_uniques()

    for model_type in df["model_type"].unique():
        filtered_df = df.query(f"model_type=='{model_type}'")
        g = sns.catplot(
            data=filtered_df,
            x="gpu",
            hue="sample_index",
            multiple="dodge",
            y="count",
            kind="bar",
        )
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(model_type)
        st.pyplot(g)


def load_converted_results(path: str):
    with open(path) as f:
        result_dict = json.load(f)
        to_convert = ["prediction", "distribution_predictions"]

        for key in to_convert:
            values = metadata.convert_json_to_np(result_dict[key])
            if len(values.shape) > 2:
                values = values.reshape((values.shape[0], -1))
            result_dict[key] = values

        # clean gpu name
        result_dict["device"]["physical_description"]["name"] = metadata.clean_gpu_name(
            result_dict["device"]["physical_description"]["name"]
        )

        return result_dict


def build_histogram(model_type: str):
    results = [load_converted_results(path) for path in get_paths(model_type)]

    df = pd.DataFrame(
        columns=["gpu", "hostname", "equivalence_class", "count", "sample_index"]
    )

    rows = []
    for result in results:
        predictions = result["distribution_predictions"]
        hostname = result["hostname"]
        gpu = metadata.clean_gpu_name(result["device"]["physical_description"]["name"])
        sample_index = result["distribution_sample_index"]

        _, counts = np.unique(predictions, return_counts=True, axis=0)
        counts = np.sort(counts)[::-1]

        for eq_class, count in enumerate(counts):
            rows.append(
                {
                    "gpu": gpu,
                    "hostname": hostname,
                    "equivalence_class": eq_class,
                    "count": count,
                    "sample_index": sample_index,
                }
            )

    df = pd.DataFrame(rows)
    try:
        return df.sort_values(["gpu", "hostname"])
    except KeyError:
        return None


def plot_histograms():
    for model_type in MODEL_TYPES:
        df = build_histogram(model_type)
        if df is None:
            continue

        g = sns.catplot(
            data=df,
            col="hostname",
            row="gpu",
            x="equivalence_class",
            y="count",
            hue="sample_index",
            kind="bar",
            aspect=3,
        )
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(model_type)
        st.pyplot(g)

        target_dir = os.path.join("results", "eval", "deterministic_cuda")
        os.makedirs(target_dir, exist_ok=True)
        plt.savefig(os.path.join(target_dir, f"{model_type}.png"))


def generate_pca(model_type):
    # load all result files
    results = [load_converted_results(path) for path in get_paths(model_type)]

    sample_indices = set([result["distribution_sample_index"] for result in results])
    dfs = []

    for sample_index in sample_indices:
        filtered_results = list(
            filter(lambda r: r["distribution_sample_index"] == sample_index, results)
        )
        # fit PCA over ALL distribution_predictions
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

        # transform predictions per GPU
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
                    "gpu": result["device"]["physical_description"]["name"],
                    "model_type": result["model_type"],
                    "hostname": result["hostname"],
                    "sample_index": result["distribution_sample_index"],
                }
            )
            dfs.append(current_df)

    try:
        full_df = pd.concat(dfs)
        return full_df.sort_values(["model_type", "gpu", "hostname"])
    except (KeyError, ValueError):
        return None


def get_paths(model_type):
    return glob.glob(
        join("results", "secondary", "deterministic_cuda", f"*{model_type}*.json")
    )


def plot_pca():
    for model_type in MODEL_TYPES:
        df = generate_pca(model_type)
        if df is None:
            continue

        g = sns.relplot(
            data=df,
            x="x",
            y="y",
            # hue="sample_index",
            col="sample_index",
            hue="hostname",
            size="count",
            style="gpu",
        )
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(model_type)
        st.pyplot(g)


def determinism_matrix():
    uniques = load_uniques()

    for model_type in MODEL_TYPES:
        df = uniques[uniques["model_type"] == model_type]

        # aggregate over sample_index
        df = df.groupby(["gpu", "hostname"]).mean().reset_index()

        pivot = df.pivot("gpu", "hostname", "count")
        fig = plt.figure()
        sns.heatmap(pivot, vmin=1, vmax=10)
        plt.title(model_type)
        st.pyplot(fig)


def visualize():
    st.markdown("## GPU determinism")

    st.markdown(
        """
### How random are the results?

Below is the number of unique predictions by GPU, as histograms
    """
    )
    # st.dataframe(load_uniques())
    plot_histograms()

    st.markdown(
        """
### Scatterplots

These scatterplots are a measure of the randomness.
They are generated by PCA embedding the 100 predictions per sample_index, each by themselves.
Therefore the location cannot be used to compare equivalence classes across sample_indices.
"""
    )
    plot_pca()


#     st.markdown(
#         """
# ### Heatmaps

# These heatmaps can be used to as a quick overview over the randomness per model type.
#         """
#     )
#     determinism_matrix()


if __name__ == "__main__":
    visualize()
