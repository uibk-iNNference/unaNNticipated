import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
import json
from innfrastructure import compare, metadata
import seaborn as sns
import streamlit as st


def get_paths():
    return glob.glob("results/main/*gpu*.json")


def load_converted_results(path):
    with open(path) as f:
        result_dict = json.load(f)
        values = metadata.convert_json_to_np(result_dict["distribution_predictions"])
        if len(values.shape) > 2:
            values = values.reshape((values.shape[0], -1))
        result_dict["distribution_predictions"] = values
        return result_dict


def get_equivalence_classes(result_dict):
    comparison_dict = {}
    for sample_index, prediction in enumerate(result_dict["distribution_predictions"]):
        comparison_dict[sample_index] = prediction

    comparison_df = compare.generate_comparison({"cifar10": comparison_dict})
    comparison_df.pop("Experiment")
    comparison_df = comparison_df.T.rename(columns={0: "class"})
    comparison_df["x"] = range(len(comparison_df))

    return comparison_df


def generate_comparison_df():
    comparisons = []
    for path in get_paths():
        df = {}
        result_dict = load_converted_results(path)

        df = get_equivalence_classes(result_dict)
        df["hostname"] = result_dict["hostname"]
        df["model_type"] = result_dict["model_type"]
        df["sample_index"] = result_dict["distribution_sample_index"]

        df["device"] = metadata.get_clean_device_name(result_dict)

        comparisons.append(df)

    comparison_df = pd.concat(comparisons)
    return comparison_df


def visualize():
    st.markdown(
        """
    ## Location of nondeterministic results

    We want to test whether the measurements within runs are truly independent.
    For this we created the plot shown below, plotting the run on the X axis, and the resulting equivalence class on the Y axis.
    Equivalence classes are calculated only within each measurement run, i.e., every line starts from equivalence class 0.
    """
    )

    df = generate_comparison_df()
    df = df.query("sample_index == 0")

    g = sns.relplot(
        data=df,
        x="x",
        y="class",
        row="model_type",
        col="device",
        hue="hostname",
        kind="line",
        aspect=2,
    )

    st.pyplot(g)

    target_dir = os.path.join("results", "eval", "location")
    os.makedirs(target_dir, exist_ok=True)
    plt.savefig(os.path.join(target_dir, f"all.png"))


if __name__ == "__main__":

    visualize()
