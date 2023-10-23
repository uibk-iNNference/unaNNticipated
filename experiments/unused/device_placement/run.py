import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from os.path import join
from joblib import Parallel, delayed, parallel_backend
from typing import List

import numpy as np
from fabric import Connection
from innfrastructure import metadata, remote
from innfrastructure.metadata import convert_json_to_np, convert_np_to_json
from invoke import run

RESULT_DIR = join("results", "device_placement")


GPU_CONFIGS = [
    "nvidia-t4",
    "nvidia-k80",
]

@dataclass
class ModelConfig:
    sample_path: str
    sample_indices: List[int]

MODEL_CONFIGS = {
    "cifar10_small": "data/datasets/cifar10/samples.npy",

}
for size in ["small", "medium", "large"]:
    model_name = f"cifar10_{size}"
    MODEL_CONFIGS[model_name] = ModelConfig("data/datasets/cifar10/samples.npy", [0])


def assumed_deterministic(_: str) -> bool:
    return False
    # gpu_name = gpu_name.lower()
    #
    # if "rtx" in gpu_name:
    #     return False
    #
    # if "1650" in gpu_name:
    #     return False
    #
    # return True


def run(host: str):
    for model_type, model_config in MODEL_CONFIGS.items():
        sample_path = model_config.sample_path

        for sample_index in model_config.sample_indices:
            result = placement_experiment(host, model_type, sample_path, sample_index)
            if result is None:
                continue

            commit = metadata.get_commit()
            result["controlling_commit"] = commit

            gpu_name = result["device"]["physical_description"]["name"]

            time_stamp = datetime.now()
            time_stamp = str(time_stamp).replace(" ", "_")

            target_path = join(
                RESULT_DIR,
                f"{host}-{model_type}-{gpu_name.replace(' ', '_')}-i_{sample_index}-{time_stamp}.json",
            )

            print(f"Saving result for {host} at {target_path}")
            with open(target_path, "w") as result_file:
                json.dump(result, result_file)


def placement_experiment(
        host: str, model_type: str, sample_path: str, sample_index: int
):
    connection = Connection(host)

    # take first sample
    samples = np.load(sample_path)
    sample = samples[sample_index]

    sample = np.expand_dims(sample, 0)
    _, path = tempfile.mkstemp(
        suffix=".npy",
        prefix=f"{host}-{model_type}-{sample_index}",
    )
    print(f"Saving the generated sample at {path}")
    np.save(path, sample)
    connection.put(local=path, remote=path)

    cmd = " ".join(
        [
            "$HOME/miniconda3/bin/conda run -n forennsic",
            f"innfcli predict {model_type} {path} -d {DEVICE} -b -l",
        ]
    )

    with connection.cd("~/Projects/forennsic/experiments"):
        result = connection.run(cmd, hide=True)
        result_dict = json.loads(result.stdout)

        placements = filter_placement(result.stderr)
        result_dict["placements"] = placements

    return result_dict


def filter_placement(stderr: str):
    return [line[30:] for line in stderr.split("\n") if "Executing op" in line]


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
