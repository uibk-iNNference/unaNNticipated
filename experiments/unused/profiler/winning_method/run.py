from innfrastructure import remote
from joblib import Parallel, delayed, parallel_backend
import tempfile
import numpy as np
from os.path import join
from datetime import datetime
from glob import glob
import os
import invoke
import gzip
from typing import Dict

from innfrastructure import metadata
from fabric.connection import Connection
import json


NUM_DATA_POINTS = 100
RESULT_DIR = join("results", "secondary", "profiler", "winning_method")


DEVICE = "/device:GPU:0"
MODEL_CONFIGS = {
    "minimal": "data/datasets/minimal/samples.npy",
}

# for size in ["small"]:
#     # for size in ["small", "medium", "large"]:
#     model_name = f"cifar10_{size}"
#     MODEL_CONFIGS[model_name] = "data/datasets/cifar10/samples.npy"

FUNCTION_NAMES = [
    "implicit_convolve_sgemm",
    "explicit_convolve_sgemm",
    "gemv2T_kernel_val",  # always FFT
    "generateWinogradTilesKernel",
    "gemv2N_kernel",
    "conv2d_grouped_direct_kernel",
]

VARIANTS = {"fft2d_r2c": "FFT_gemv2N", "winogradForwardFilter": "winograd_nonfused"}

# hosts = ["rechenknecht"]
hosts = ["ennclave-server", "rechenknecht", "ennclave-consumer"]


def predict_data_points(connection, host, path, model_type):
    results = []
    current_dict = {}
    unique_predictions = {}

    def _get_equivalence_class(current_prediction):
        try:
            return unique_predictions[current_prediction]
        except KeyError:
            if len(unique_predictions) == 0:
                current_max = -1
            else:
                max_tuple = max(unique_predictions.items(), key=lambda x: x[1])
                current_max = max_tuple[1]

            equivalence_class = current_max + 1
            unique_predictions[current_prediction] = equivalence_class

            return equivalence_class

    logdir = tempfile.mkdtemp(prefix=f"{host}-{model_type}-")
    cmd = " ".join(
        [
            "$HOME/miniconda3/bin/conda run -n forennsic",
            f"innfcli predict {model_type} {path} -d {DEVICE} -b -p {logdir}",
        ]
    )

    for i in range(NUM_DATA_POINTS):
        if i % 10 == 0:
            print(f"Distribution run {i+1} on {host}: {cmd}")

        experiment_time_stamp = datetime.now()
        experiment_time_stamp = str(experiment_time_stamp).replace(" ", "_")

        # delete existing logs (-f ignores errors if doesn't exist)
        connection.run(f"rm -rf {logdir}")

        # get results
        result_stdout = connection.run(cmd, hide=True).stdout
        current_dict = json.loads(result_stdout)
        prediction = current_dict["prediction"]
        device = metadata.get_clean_device_name(current_dict)

        # only after first iteration
        if i == 0:
            local_logdir = f"logs-{host}-{model_type}-{device}"
            os.makedirs(join(RESULT_DIR, local_logdir), exist_ok=True)

        # get winning method
        trace_path = download_logs(host, logdir)
        winning_algorithm = convert_trace(
            trace_path, experiment_time_stamp, local_logdir
        )
        print(winning_algorithm)

        # get equivalence class
        result_string = prediction["bytes"]
        equivalence_class = _get_equivalence_class(result_string)

        results.append(
            {
                "prediction": current_dict["prediction"],
                "equivalence_class": equivalence_class,
                "winning_algorithm": winning_algorithm,
                "time_stamp": experiment_time_stamp,
            }
        )

    return results, current_dict


def convert_trace(path: str, experiment_time_stamp, local_logdir) -> Dict:
    # save filtered traces to txt
    # parse trace
    with gzip.open(path, "r") as unzipped:
        raw_trace = json.load(unzipped)

    events = raw_trace["traceEvents"]
    events = filter(
        lambda e: "ts" in e.keys()
        and not (e["name"].startswith("Memory") or e["name"].startswith("$")),
        events,
    )
    events = sorted(events, key=lambda e: e["ts"])

    target_log_filename = f"{experiment_time_stamp}.txt"
    target_log_path = join(RESULT_DIR, local_logdir, target_log_filename)

    winning_algorithm = ""
    last_variant = ""
    with open(target_log_path, "w") as filtered_trace_logs:
        for event in events:
            name = event["name"]
            filtered_trace_logs.write(name + "\n")

            for function_name in FUNCTION_NAMES:
                if function_name in name:
                    winning_algorithm = function_name

            for variant in VARIANTS:
                if variant in name:
                    last_variant = variant

    if winning_algorithm == "gemv2T_kernel_val":
        winning_algorithm = "FFT_gemv2T"

    if winning_algorithm == "gemv2N_kernel":
        winning_algorithm = VARIANTS[last_variant]

    return winning_algorithm


def download_logs(host: str, logdir: str) -> str:
    """Download traces from remote host, and return the path to the gzipped json file on local

    Args:
        host (str): The remote hostname
        logdir (str): The log directory on the remote (as upassed to innfcli)

    Returns:
        str: the path to the file on the local machine
    """
    basedir = os.path.dirname(logdir)
    invoke.run(f"rm -rf {logdir}")
    print(f"Downloading remote logdir {logdir} to {logdir}")
    invoke.run(f"rsync -avz {host}:'{logdir}' '{basedir}/'")
    glob_term = join(logdir, "**", "*trace.json.gz")
    traces = glob(glob_term, recursive=True)

    assert len(traces) == 1
    return traces[0]


def run(host: str):

    for model_type, sample_path in MODEL_CONFIGS.items():
        connection = Connection(host)

        samples = np.load(sample_path)
        sample = samples[0]

        sample = np.expand_dims(sample, 0)
        _, path = tempfile.mkstemp(suffix=".npy", prefix=f"{host}-{model_type}-")

        print(f"Saving the generated sample at {path}")
        np.save(path, sample)
        connection.put(local=path, remote=path)

        with connection.cd("~/Projects/forennsic/experiments"):
            results, raw_dict = predict_data_points(connection, host, path, model_type)

        result_dict = {
            "results": results,
            "hostname": host,
        }
        copy_keys = [
            "cpu_info",
            "device",
            "model_type",
            "batch_dissemination",
            "executing_commit",
        ]
        for copy_key in copy_keys:
            result_dict[copy_key] = raw_dict[copy_key]

        result_dict["controlling_commit"] = metadata.get_commit()

        cleaned_host = host.replace("-", "_")

        target_filename = f"{cleaned_host}-{model_type}-{metadata.get_clean_device_name(raw_dict)}.json"
        target_path = join(RESULT_DIR, target_filename)

        print(f"Saving result for {host} at {target_path}")
        with open(target_path, "w") as result_file:
            json.dump(result_dict, result_file)


def ensure_results_dir():
    os.makedirs(RESULT_DIR, exist_ok=True)


def main():
    with parallel_backend("loky"):
        Parallel()(delayed(remote.prepare_server)(host) for host in hosts)

    ensure_results_dir()

    # run(hosts[0])
    with parallel_backend("loky"):
        Parallel()(delayed(run)(host) for host in hosts)


if __name__ == "__main__":
    main()
