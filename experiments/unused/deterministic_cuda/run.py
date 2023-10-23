#!/usr/bin/env python3
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from os.path import join
from threading import Thread
from typing import List

import numpy as np
from fabric import Connection
from innfrastructure import metadata, remote
from innfrastructure.metadata import convert_json_to_np, convert_np_to_json
from joblib import Parallel, delayed, parallel_backend

hosts = ["ennclave-server", "rechenknecht", "ennclave-consumer"]


NUM_ND_DATA_POINTS = 100
RESULT_DIR = join("results", "secondary", "deterministic_cuda")
DEVICE = "/device:GPU:0"


@dataclass
class ModelConfig:
    sample_path: str
    sample_indices: List[int]


MODEL_CONFIGS = {
    "minimal": ModelConfig("data/datasets/minimal/samples.npy", [0]),
}

for size in ["small", "medium", "large"]:
    model_name = f"cifar10_{size}"
    MODEL_CONFIGS[model_name] = ModelConfig("data/datasets/cifar10/samples.npy", [0])


def run(host):
    # Instructions on how to make results more reproducible taken from [here](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility)
    experiment_per_host(host, ":4096:8")
    experiment_per_host(host, ":16:8")


def experiment_per_host(host: str, cublas_workspace_config: str):
    for model_type, model_config in MODEL_CONFIGS.items():
        sample_path = model_config.sample_path

        for sample_index in model_config.sample_indices:
            result = nondeterministic_experiment(
                host, model_type, sample_path, sample_index, cublas_workspace_config
            )
            if result is None:
                continue

            commit = metadata.get_commit()
            result["controlling_commit"] = commit

            time_stamp = datetime.now()
            time_stamp = str(time_stamp).replace(" ", "_")

            cleaned_host = host.replace("-", "_")

            gpu_name = result["device"]["physical_description"]["name"]
            gpu_name = metadata.clean_gpu_name(gpu_name)

            target_path = join(
                RESULT_DIR,
                f"{cleaned_host}-{model_type}-gpu_{gpu_name.replace(' ', '_')}-cublas_{cublas_workspace_config}-i_{sample_index}-{time_stamp}.json",
            )

            print(f"Saving result for {host} at {target_path}")
            with open(target_path, "w") as result_file:
                json.dump(result, result_file)


def nondeterministic_experiment(
    host: str,
    model_type: str,
    sample_path: str,
    sample_index: int,
    cublas_workspace_config: str,
):
    connection = Connection(host)

    # take sample at sample index
    samples = np.load(sample_path)
    sample = samples[sample_index]

    sample = np.expand_dims(sample, 0)
    _, path = tempfile.mkstemp(
        suffix=".npy",
        prefix=f"{host}-{model_type}-i_{sample_index}-",
    )
    print(f"Saving the generated sample at {path}")
    np.save(path, sample)
    connection.put(local=path, remote=path)

    cmd = " ".join(
        [
            "$HOME/miniconda3/bin/conda run -n forennsic",
            f"innfcli predict {model_type} {path} -d {DEVICE} -b --cublas_workspace_config {cublas_workspace_config}",
        ]
    )

    with connection.cd("~/Projects/forennsic/experiments"):
        result_dict, results = predict_sample_distribution(connection, cmd, host)

    result_dict["deterministic"] = False
    result_dict["distribution_sample_index"] = sample_index
    result_dict["distribution_predictions"] = convert_np_to_json(results)
    return result_dict


def predict_sample_distribution(connection, cmd, host):
    results = []
    result_dict = {}

    for i in range(NUM_ND_DATA_POINTS):
        if i % 10 == 0:
            print(f"Distribution run {i+1} on {host}: {cmd}")

        result_stdout = connection.run(cmd, hide=True).stdout
        result_dict = json.loads(result_stdout)
        result_np = convert_json_to_np(result_dict["prediction"])
        results.append(result_np)

    return result_dict, np.vstack(results)


def main():
    with parallel_backend("loky"):
        Parallel()(delayed(remote.prepare_server)(host) for host in hosts)

    ensure_results_dir()

    with parallel_backend("loky"):
        Parallel()(delayed(run)(host) for host in hosts)


def ensure_results_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)


if __name__ == "__main__":
    main()
