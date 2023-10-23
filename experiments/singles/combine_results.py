import json
import pandas as pd
from innfrastructure import metadata
import glob
from os.path import join
import os

output_configs = {
    "cifar10_small": 10,
    "cifar10_medium": 10,
    "cifar10_large": 10,
    "small": 5,
}


def parse_gpu_result(path):
    with open(path, "r") as f:
        results_dict = json.load(f)

    # build df from json dict
    predictions = metadata.convert_json_to_np(results_dict["distribution_predictions"])
    predictions = predictions.reshape((predictions.shape[0], -1))

    df_values = {}
    for i in range(10):
        df_values[f"p_{i}"] = predictions[:, i]

    df_values["gpu_name"] = results_dict["device"]["physical_description"]["name"]
    df_values["hostname"] = results_dict["hostname"]

    return pd.DataFrame(df_values)


def parse_gpu_results():
    paths = glob.glob(join("results", "gpu", "*.json"))

    for model_type, num_classes in output_configs.items():
        filtered_paths = filter(lambda p: model_type in p, paths)
        gpu_dfs = [parse_gpu_result(path) for path in filtered_paths]
        combined_df = pd.concat(gpu_dfs)

        target_dir = join("results", "combined")
        filename = f"gpu_{model_type}_combined.csv"
        os.makedirs(target_dir, exist_ok=True)
        combined_df.to_csv(join(target_dir, filename))


if __name__ == "__main__":
    parse_gpu_results()
