from glob import glob
from io import BytesIO
import json
import logging, os, tempfile
from fabric import Connection
from tensorflow import keras
import numpy as np
from innfrastructure import metadata, models, InnfrastructureConnection
import click
from joblib import Parallel, parallel_backend, delayed

MODEL_CONFIGS = [
    "cifar10_medium_sigmoid",
    "cifar10_medium_sigmoid_untrained",
    "cifar10_small",
    "cifar10_small_sigmoid",
    "cifar10_small_sigmoid_untrained",
]
SAMPLE_PATH = os.path.join("data", "datasets", "cifar10", "samples.npy")
SAMPLE_INDICES = [0]
RESULT_DIR = os.path.join("results", "instrumental")


def prepare(model_dir: str, target_model: str) -> str:
    logging.info(f"Building instrumental for {target_model}")

    # get model and layer
    model = models.get_model(target_model)
    instrumental_model = keras.models.Model(
        model.inputs[0],
        [layer.output for layer in model.layers],
        name=model.name + "_instrumental",
    )
    instrumental_model.save(os.path.join(model_dir, f"{instrumental_model.name}.h5"))


def cpu_experiment(
    hostname: str,
    model_dir: str,
):
    connection = InnfrastructureConnection(hostname)

    instrumental_model_paths = glob(os.path.join(model_dir, "*.h5"))
    for model_path in instrumental_model_paths:
        connection.run(f"mkdir -p {model_dir}")
        connection.put(model_path, model_path)
    connection.docker_run(f"mv {model_dir}/*.h5 /forennsic/data/models")

    for model_type in MODEL_CONFIGS:
        full_model_type = model_type + "_instrumental"
        print(f"Performing CPU experiment for {full_model_type} on {hostname}...")

        samples = np.load(SAMPLE_PATH)
        cmd = f"innfcli predict {full_model_type} /tmp/sample.npy -b -d /device:CPU:0"

        for sample_index in SAMPLE_INDICES:
            current_sample = samples[sample_index : sample_index + 1]
            f = BytesIO()
            np.save(f, current_sample)
            print(f"Sending sample number {sample_index} to host {hostname}...")
            connection.put(f, "/tmp/sample.npy")

            print(f"Running prediction {full_model_type} i{sample_index} on {hostname}")
            result = json.loads(connection.docker_run(cmd, hide=True).stdout)

            save_result(hostname, model_type, result, RESULT_DIR, sample_index)


def save_result(hostname: str, model_type: str, result, result_dir, sample_index):
    cleaned_host = hostname.replace("-", "_")
    device_name = metadata.get_clean_device_name(result)
    clean_model_type = (
        model_type
        if model_type.endswith("_instrumental")
        else model_type + "_instrumental"
    )
    target_dir = os.path.join(
        result_dir,
        f"{clean_model_type}-i{sample_index}",
    )
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(
        target_dir,
        f"{cleaned_host}-{device_name}.json",
    )
    with open(target_path, "w") as target_file:
        print(f"Saving result in {target_path}...")
        json.dump(result, target_file)


@click.command()
def main():
    print(f"Building instrumental models and inputs for {MODEL_CONFIGS}")

    # prepare directories
    tempdir = tempfile.mkdtemp(prefix="instrumental_")
    model_dir = os.path.join(tempdir, "models")
    os.makedirs(model_dir, exist_ok=True)

    for model in MODEL_CONFIGS:
        prepare(model_dir, model)

    hosts = ["rechenknecht", "server"]
    # for host in hosts:
    #     cpu_experiment(host, model_dir)

    with parallel_backend("loky"):
        Parallel()(delayed(cpu_experiment)(host, model_dir) for host in hosts)


if __name__ == "__main__":
    main()
