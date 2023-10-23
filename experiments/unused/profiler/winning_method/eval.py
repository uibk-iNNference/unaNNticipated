from typing import List
from os.path import join
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
    # "cifar10_small",
    # "cifar10_medium",
    # "cifar10_large",
    "minimal",
]


def load_converted_results(path: str):
    with open(path) as f:
        result_dict = json.load(f)
        results = result_dict["results"]
        to_convert = ["prediction"]

        for i, entry in enumerate(results):
            for key in to_convert:
                original_value = entry[key]
                values = metadata.convert_json_to_np(original_value)
                if len(values.shape) > 2:
                    values = values.reshape((values.shape[0], -1))
                entry[key] = values
                entry[key + "_raw"] = original_value

        # clean gpu name
        result_dict["device_name"] = metadata.get_clean_device_name(result_dict)

        return result_dict


def generate_pca(model_type):
    # load all result files
    results = [load_converted_results(path) for path in get_result_paths(model_type)]

    dfs = []

    # fit PCA over ALL distribution_predictions
    pca_data = [_extract_predictions(result) for result in results]
    pca_data = np.vstack(pca_data)

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
    for result in results:
        equivalence_class = {
            "x": "transformed[x]",
            "y": "transformed[y]",
            "winning_algorithm": "winning_algorithm",
        }

        equivalence_classes = {}
        for prediction in result["results"]:
            equivalence_class = prediction["equivalence_class"]
            winning_algorithm = prediction["winning_algorithm"]

            try:
                current_dict = equivalence_classes[equivalence_class]
                current_dict["count"] += 1
                current_dict["winning_algorithms"].add(winning_algorithm)
            except KeyError:
                transformed_coordinates = pca.transform(prediction["prediction"])
                x = transformed_coordinates[0, 0]
                y = transformed_coordinates[0, 1]

                new_dict = {
                    "x": x,
                    "y": y,
                    "winning_algorithms": set([winning_algorithm]),
                    "count": 1,
                }
                equivalence_classes[equivalence_class] = new_dict

        rows = []
        for equivalence_class, values in equivalence_classes.items():
            algorithm_string = ", ".join(values["winning_algorithms"])
            rows.append(
                {
                    "equivalence_class": equivalence_class,
                    "x": values["x"],
                    "y": values["y"],
                    "count": values["count"],
                    "winning_algorithms": algorithm_string,
                    "hostname": result["hostname"],
                    "device": result["device_name"],
                }
            )
        current_df = pd.DataFrame(rows)
        dfs.append(current_df)

    full_df = pd.concat(dfs)
    return full_df


def _extract_predictions(results_dict):
    data = []
    for prediction in results_dict["results"]:
        data.append(prediction["prediction"])
    data = np.vstack(data)
    return data


def get_result_paths(model_type) -> List[str]:
    return glob.glob(
        join(
            "results",
            "secondary",
            "profiler",
            "winning_method",
            f"*{model_type}*.json",
        )
    )


def get_log_paths(model_type) -> List[str]:
    return glob.glob(
        join(
            "results",
            "secondary",
            "profiler",
            "winning_method",
            f"logs-{model_type}*",
            "*.txt",
        )
    )


def plot_pca():
    for model_type in MODEL_TYPES:
        df = generate_pca(model_type)

        g = sns.relplot(
            data=df,
            x="x",
            y="y",
            # hue="sample_index",
            col="hostname",
            hue="winning_algorithms",
            size="count",
            row="device",
        )
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(model_type)
        st.pyplot(g)


def get_equivalence_class_consistency(model_type: str) -> pd.DataFrame:
    def _is_consistent(result):
        algorithm_classes = {}
        for prediction in result["results"]:
            winning_algorithm = prediction["winning_algorithm"]
            equivalence_class = prediction["equivalence_class"]
            try:
                previous_class = algorithm_classes[winning_algorithm]
                if equivalence_class != previous_class:
                    return False
            except KeyError:
                algorithm_classes[winning_algorithm] = equivalence_class

        return True

    results = [load_converted_results(path) for path in get_result_paths(model_type)]

    rows = []
    for result in results:
        rows.append(
            {
                "hostname": result["hostname"],
                "device": result["device_name"],
                "consistent": "consistent"
                if _is_consistent(result)
                else "INconsistent",
            }
        )

    return pd.DataFrame(rows)


def visualize_algorithm_consistency(model_type: str):
    df = get_equivalence_class_consistency(model_type)
    pivot = df.pivot(index="hostname", columns="device", values="consistent")

    st.dataframe(pivot)


def get_algorithms_per_equivalence_class(model_type: str) -> pd.DataFrame:
    results = [load_converted_results(path) for path in get_result_paths(model_type)]
    rows = []

    for result in results:
        algorithms = {}
        for prediction in result["results"]:
            current_class = prediction["equivalence_class"]
            algorithm = prediction["winning_algorithm"]
            try:
                algorithms[current_class].add(algorithm)
            except KeyError:
                algorithms[current_class] = set([algorithm])

        for equivalence_class, algorithm_set in algorithms.items():
            rows.append(
                {
                    "identifier": result["hostname"] + result["device_name"],
                    "equivalence_class": equivalence_class,
                    "algorithms": "\n".join(algorithm_set),
                }
            )

    return pd.DataFrame(rows)


def visualize_algorithms_per_equivalence_class(model_type: str):
    df = get_algorithms_per_equivalence_class(model_type)
    pivot = df.pivot(
        index="identifier", columns="equivalence_class", values="algorithms"
    )

    st.dataframe(pivot)


def get_global_equivalence_classes(model_type: str) -> pd.DataFrame:
    results = [load_converted_results(path) for path in get_result_paths(model_type)]
    rows = []
    equivalence_classes = {}
    next_equivalence_class = 0

    for result in results:
        algorithms = {}

        for prediction in result["results"]:
            byte_string = prediction["prediction_raw"]["bytes"]
            try:
                equivalence_class = equivalence_classes[byte_string]
            except KeyError:
                equivalence_classes[byte_string] = next_equivalence_class
                equivalence_class = next_equivalence_class
                next_equivalence_class += 1

            current_algorithm = prediction["winning_algorithm"]
            try:
                algorithms[equivalence_class].add(current_algorithm)
            except KeyError:
                algorithms[equivalence_class] = set([current_algorithm])

        for equivalence_class, algorithm_set in algorithms.items():
            rows.append(
                {
                    # "identifier": result["hostname"] + result["device_name"],
                    "hostname": result["hostname"],
                    "device_name": result["device_name"],
                    "equivalence_class": equivalence_class,
                    "algorithms": "\n".join(algorithm_set),
                }
            )

    return pd.DataFrame(rows)


def get_global_l2norm(model_type: str) -> pd.DataFrame:
    results = [load_converted_results(path) for path in get_result_paths(model_type)]
    rows = []
    equivalence_classes = {}
    next_equivalence_class = 0
    base_vect = []

    l2_norms = {}
    for result in results:
        for prediction in result["results"]:
            prediction_raw = prediction["prediction_raw"]
            byte_string = prediction_raw["bytes"]
            current_result = metadata.convert_json_to_np(prediction_raw)

            try:
                equivalence_class = equivalence_classes[byte_string]
            except KeyError:
                equivalence_classes[byte_string] = next_equivalence_class
                equivalence_class = next_equivalence_class
                next_equivalence_class += 1

            if equivalence_class == 0:
                base_vect = metadata.convert_json_to_np(prediction_raw)

            norm = np.sum(np.power((current_result - base_vect), 2))

            l2_norms[equivalence_class] = norm

    for equivalence_class, norm in l2_norms.items():
        rows.append(
            {
                "equivalence_class": equivalence_class,
                "L2_norms": norm,
            }
        )
    return pd.DataFrame(rows)


def visualize_global_equivalence_classes(model_type: str):
    df = get_global_equivalence_classes(model_type)
    pivot = df.pivot(
        index=["hostname", "device_name"],
        columns="equivalence_class",
        values="algorithms",
    )
    pivot = pivot.reset_index()

    st.dataframe(pivot)


def visualize_L2_norm(model_type: str):
    df = get_global_l2norm(model_type)
    pivot = df.pivot(
        columns="equivalence_class",
        values="L2_norms",
    )
    pivot = pivot.reset_index()

    st.dataframe(pivot)


def visualize():
    st.markdown(
        "## GPU equivalence classes per sample as function of the 'Winning Algorithm'"
    )

    st.markdown(
        """
# Scatterplots

Here we plot a PCA fitting of the predictions, and identify them based on the winning algorithm
"""
    )
    plot_pca()

    st.markdown(
        """
# Equivalence classes with winning algorithms

This table lists the equivalence classes of the results in combination with the respective winning algorithm.
        """
    )
    st.markdown(
        """
Let's check out the L2-Norm Distribution of our Predictions
    """
    )
    visualize_L2_norm("minimal")

    st.markdown(
        """
First we test whether winning algorithms occur in more than one class:
    """
    )
    visualize_algorithm_consistency("minimal")

    st.markdown(
        """
Next we examine which algorithms produce which equivalence classes (equivalence classes are computed anew per combination of host and GPU)
    """
    )
    visualize_algorithms_per_equivalence_class("minimal")

    st.markdown(
        """
Next we examine the algorithms if we take equivalence classes over ALL machines and GPUs (i.e. two results in the same equivalence class are EXACTLY identical, even on different machines and different hardware)
    """
    )
    visualize_global_equivalence_classes("minimal")


if __name__ == "__main__":
    visualize()
