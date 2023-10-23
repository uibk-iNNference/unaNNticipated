#!/usr/bin/env python3

from datetime import datetime
from joblib import Parallel, parallel_backend, delayed
import os
from os.path import join
import tempfile
from fabric.connection import Connection
import invoke
from joblib.parallel import delayed
import tensorflow as tf
import numpy as np
import json

from innfrastructure import metadata, remote

hosts = ["ennclave-server", "rechenknecht", "ennclave-consumer"]

RESULT_LOG_DIR = join("results", "secondary", "profiler", "logs_heuristic")
RESULT_DIR = join("results", "secondary", "profiler", "log_samples_heuristic")
DEVICES = ["/device:CPU:0", "/device:GPU:0"]


MODEL_CONFIGS = {
    "minimal": "data/datasets/minimal/samples.npy",
}

for size in ["small", "medium", "large"]:
    model_name = f"cifar10_{size}"
    MODEL_CONFIGS[model_name] = "data/datasets/cifar10/samples.npy"


def run(host, run_cpu, run_gpu):
    if run_cpu:
        experiment_per_device(host, "/device:CPU:0")
    if run_gpu:
        experiment_per_device(host, "/device:GPU:0")


def experiment_per_device(host: str, device: str):
    for model_type, sample_path in MODEL_CONFIGS.items():
        result, logs = experiment(host, model_type, sample_path, device)

        if result is None:
            continue

        result["controlling_commit"] = metadata.get_commit()

        time_stamp = datetime.now()
        time_stamp = str(time_stamp).replace(" ", "_")

        cleaned_host = host.replace("-", "_")

        # if device == gpu
        try:
            gpu_name = result["device"]["physical_description"]["name"]
            gpu_name = metadata.clean_gpu_name(gpu_name)
            cleaned_gpu = gpu_name.replace(" ", "_")

            target_result_path = join(
                RESULT_DIR,
                f"{cleaned_host}-{model_type}-gpu_{cleaned_gpu}-profiler-{time_stamp}.json",
            )
            target_log_path = join(
                RESULT_LOG_DIR,
                f"{cleaned_host}-{model_type}-gpu_{cleaned_gpu}-profiler-{time_stamp}.json",
            )
        except KeyError:
            target_result_path = join(
                RESULT_DIR,
                f"{cleaned_host}-{model_type}-cpu-profiler-{time_stamp}.json",
            )
            target_log_path = join(
                RESULT_LOG_DIR,
                f"{cleaned_host}-{model_type}-cpu-profiler-{time_stamp}.json",
            )

        print(f"Saving result for {host} at {target_result_path}")
        print(f"Saving logs for {host} at {target_log_path}")

        with open(target_result_path, "w") as result_file:
            json.dump(result, result_file)

        with open(target_log_path, "w") as log_file:
            json.dump(logs, log_file)


def experiment(host: str, model_type: str, sample_path: str, device: str):
    connection = Connection(host)

    # take sample at sample index 0
    samples = np.load(sample_path)
    sample = samples[0:2]
    # sample = np.expand_dims(sample, 0)

    _, path = tempfile.mkstemp(
        suffix=".npy",
        prefix=f"{host}-{model_type}-",
    )
    print(f"Saving the generated sample at {path}")
    np.save(path, sample)
    connection.put(local=path, remote=path)

    remote_logdir = tempfile.TemporaryDirectory().name

    cmd = " ".join(
        [
            "$HOME/miniconda3/bin/conda run -n forennsic",
            f"innfcli predict {model_type} {path} -d {device} -b -p {remote_logdir}",
        ]
    )

    with connection.cd("~/Projects/forennsic/experiments"):
        log_stdout = connection.run(cmd, hide=True).stdout
        log_dict = json.loads(log_stdout)
        result_dict, results = predict_sample(connection, cmd)

    result_dict["prediction"] = metadata.convert_np_to_json(results)

    host = result_dict["hostname"]
    cleaned_host = host.replace("-", "_")
    model_type = result_dict["model_type"]
    device_name = metadata.get_clean_device_name(result_dict).replace(" ", "_")

    local_logdir = f"{cleaned_host}-{device_name}-{model_type}"
    full_local_path = join(RESULT_LOG_DIR, local_logdir)

    print(
        f"Downloading remote logdir {remote_logdir} to local logdir {full_local_path}"
    )
    invoke.run(f"rsync -avz {host}:'{remote_logdir}' '{full_local_path}/'")

    return result_dict, log_dict


def predict_sample(connection, cmd):
    results = []
    result_dict = {}

    result_stdout = connection.run(cmd, hide=True).stdout
    result_dict = json.loads(result_stdout)
    result_np = metadata.convert_json_to_np(result_dict["prediction"])
    results.append(result_np)

    return result_dict, np.vstack(results)


def main(run_cpu, run_gpu):
    with parallel_backend("loky"):
        Parallel()(delayed(remote.prepare_server)(host) for host in hosts)

    ensure_results_dir()

    with parallel_backend("loky"):
        Parallel()(delayed(run)(host, run_cpu, run_gpu) for host in hosts)


def ensure_results_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(RESULT_LOG_DIR, exist_ok=True)


if __name__ == "__main__":

    import sys

    run_cpu = "--run_cpu" in sys.argv
    run_gpu = "--run_gpu" in sys.argv

    main(run_cpu, run_gpu)
