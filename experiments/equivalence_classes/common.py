from io import BytesIO
import numpy as np
import json
from posixpath import join

from fabric import Connection

from innfrastructure import metadata

MODEL_CONFIGS = [
    # don't run minimal, that comes in the winning_algorithm experiment
    ("cifar10_small", "data/datasets/cifar10/samples.npy", [0, 1, 6]),
    ("cifar10_medium", "data/datasets/cifar10/samples.npy", [0, 1, 6]),
    ("deep_weeds", "data/datasets/deep_weeds/samples.npy", [0, 6, 1]),
]


def cpu_experiment(
        connection: Connection,
        hostname: str,
        result_dir: str,
):
    for model_type, sample_path, sample_indices in MODEL_CONFIGS:
        print(f"Performing CPU experiment for {model_type} on {hostname}...")

        samples = np.load(sample_path)
        cmd = f"innfcli predict {model_type} /tmp/sample.npy -b -d /device:CPU:0"

        for sample_index in sample_indices:
            current_sample = samples[sample_index: sample_index + 1]
            f = BytesIO()
            np.save(f, current_sample)
            print(f"Sending sample number {sample_index} to host {hostname}...")
            connection.put(f, "/tmp/sample.npy")

            print(f"Running prediction {model_type} i{sample_index} on {hostname}")
            result = json.loads(connection.docker_run(cmd, hide=True).stdout)

            save_result(hostname, model_type, result, result_dir, sample_index)


def save_result(hostname, model_type, result, result_dir, sample_index):
    cleaned_host = hostname.replace("-", "_")
    device_name = metadata.get_clean_device_name(result)
    target_path = join(
        result_dir,
        f"{cleaned_host}-{device_name}-{model_type}-i{sample_index}.json",
    )
    with open(target_path, "w") as target_file:
        print(f"Saving result in {target_path}...")
        json.dump(result, target_file)


def gpu_experiment(
        connection: Connection,
        hostname: str,
        num_runs: int,
        result_dir: str,
):
    for model_type, sample_path, sample_indices in MODEL_CONFIGS:
        print(f"Performing GPU experiment for {model_type} on {hostname}...")

        samples = np.load(sample_path)
        cmd = f"innfcli predict {model_type} /tmp/sample.npy -b -d /device:GPU:0"

        for sample_index in sample_indices:

            current_sample = samples[sample_index: sample_index + 1]
            f = BytesIO()
            np.save(f, current_sample)
            print(f"Sending sample number {sample_index} to host {hostname}...")
            connection.put(f, "/tmp/sample.npy")

            predictions = []

            result = {}
            for i in range(num_runs):
                print(f"Prediction {i} on {hostname}")
                result = json.loads(connection.docker_run(cmd, hide=True).stdout)
                prediction = metadata.convert_json_to_np(
                    result["prediction"]
                )
                predictions.append(prediction)

            predictions = np.vstack(predictions)
            uniques, counts = np.unique(predictions, return_counts=True, axis=0)
            prediction, count = sorted(zip(uniques, counts), key=lambda x: x[1])[-1]
            count = int(count)
            result["all_predictions"] = metadata.convert_np_to_json(predictions)
            result["prediction"] = metadata.convert_np_to_json(prediction)
            result["count"] = count

            save_result(hostname, model_type, result, result_dir, sample_index)
