import os
import numpy as np
import streamlit as st
from innfrastructure import metadata, compare
import json
from glob import glob
import pandas as pd
from os.path import join, basename, splitext

MODEL_TYPES = ["minimal", "cifar10_small", "cifar10_medium", "cifar10_large"]

for path in glob(join("data", "models", "conv_k*_i*.h5")):
    filename = basename(path)
    MODEL_TYPES.append(splitext(filename)[0])


def get_paths():
    paths = glob("results/main/**/*conv*.json", recursive=True)
    return [p for p in paths if "contract" not in p and "instrumental" not in p]


def load_converted_results(path: str):
    with open(path) as f:
        result_dict = json.load(f)
        distribution = [
            metadata.convert_json_to_np(p)
            for p in result_dict["distribution_predictions"]
        ]
        result_dict["distribution_predictions"] = np.vstack(distribution)
        return result_dict


def visualize():
    st.markdown(
        """
        ## Equivalence classes
        
        Here you can see the equivalence classes we generate from our ressults.
        As some of our hardware has been observed as nondeterministic (mostly the RTX 2070), we use the most common prediction for the equivalence clases.
        A study of the observed randomness is available elsewhere.

        All our GPUs are either GTX or RTX GeForce consumer GPUs manufactured by Nvidia.

        The CPUs are all made by Intel.
        The i7_9700 is a 9th generation desktop processor, formerly codenamed Coffee Lake.
        The e3_1270_v6 is a server processor, formerly codenamed Kaby Lake.
        Both come from the Skylake microarchitecture.

        Our third processor is a Broadwell microarchitecture 4th generation desktop processor.
        """
    )

    paths = get_paths()
    results = [load_converted_results(path) for path in paths]

    target_dir = os.path.join("results", "eval", "equivalence_classes")

    dfs = [
        generate_equivalence_classes(results, model_type) for model_type in MODEL_TYPES
    ]
    for df, model_type in zip(dfs, MODEL_TYPES):
        df.insert(0, "model_type", model_type)
    single_df = pd.concat(dfs)
    single_df.rename(
        columns={"model_type": "Model Type", "Sample_Index": "Sample Index"},
        inplace=True,
    )
    single_df.set_index(["Model Type", "Sample Index"], inplace=True)
    single_df = single_df.T

    st.dataframe(single_df)

    os.makedirs(target_dir, exist_ok=True)
    single_df.to_markdown(os.path.join(target_dir, "all.md"))
    single_df.to_latex(os.path.join(target_dir, "equivalence-classes.tex"))


ORDER = [
    "amd-rome",
    "amd-milan",
    "intel-ice-lake",
    "intel-cascade-lake",
    "intel-skylake",
    "intel-broadwell",
    "intel-haswell",
    "intel-sandy-bridge",
    "intel-ivy-bridge",
    "nvidia-tesla-k80",
    "nvidia-tesla-t4",
    "nvidia-tesla-v100",
    "nvidia-tesla-p100",
    "nvidia-tesla-a100",
]


def find_order_index(result):
    hostname = result["hostname"]

    for i, element in enumerate(ORDER):
        if element == hostname:
            return i

    raise ValueError(f"Hostname {hostname} not found in order")


def generate_equivalence_classes(results, model_type):
    layout = {}

    model_results = list(filter(lambda r: r["model_type"] == model_type, results))

    sample_indices = set(map(lambda r: r["distribution_sample_index"], model_results))

    for index in sample_indices:
        sample_results = [
            result
            for result in model_results
            if result["distribution_sample_index"] == index
        ]

        sample_results = sorted(sample_results, key=find_order_index)
        predictions = {}

        for result in sample_results:
            identifier = " ".join(
                [word.capitalize() for word in result["hostname"].split("-")]
            )

            distribution_predictions = result["distribution_predictions"]
            unique_predictions, counts = np.unique(
                distribution_predictions, return_counts=True, axis=0
            )
            sorted_uniques = sorted(
                zip(unique_predictions, counts), key=lambda x: x[1], reverse=True
            )
            most_common_prediction = sorted_uniques[0][0]

            predictions[identifier] = most_common_prediction

        layout[str(index)] = predictions

    df = compare.generate_comparison(layout).rename(
        columns={"Experiment": "Sample_Index"}
    )
    return df


if __name__ == "__main__":
    visualize()
