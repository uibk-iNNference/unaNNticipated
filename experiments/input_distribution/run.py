import itertools
import random

random.seed(42)

# Numpy RNG
import numpy as np

np.random.seed(42)

# TF RNG
from tensorflow.python.framework import random_seed

random_seed.set_seed(42)

import json
import numpy as np
from os.path import join, basename
import invoke
from fabric import Connection, connection
import tempfile
import click
import logging
import os

from tensorflow import keras
from joblib.parallel import Parallel, parallel_backend, delayed

from innfrastructure import models, metadata, gcloud, remote

TARGET_MODEL = "cifar10_medium"
SAMPLE_INDEX = 0
TARGET_LAYER = "conv2d_11"
NUM_PERMUTATIONS = 3
RESULT_DIR = join("results", "input_distribution")


def prepare(target_model: str, target_layer: str, sample_index: str) -> str:
    logging.info(f"Extracting layer {target_layer} from {target_model}")

    # prepare target
    tempdir = tempfile.mkdtemp(prefix="input_distribution-")

    extract_samples(target_model, target_layer, sample_index, tempdir)
    extract_layer(target_model, target_layer, tempdir)

    return tempdir


def extract_samples(
    target_model: str,
    target_layer: str,
    sample_index: int,
    target_dir: str,
):
    model = models.get_model(target_model)

    # get index of layer (required for output)
    names = [layer.name for layer in model.layers]
    for i, name in enumerate(names):
        if name == target_layer:
            target_index = i

    # get an output
    target_path = join(
        "results",
        "instrumental",
        f"{target_model}_instrumental-i{sample_index}",
        f"intel_skylake.json",
    )
    with open(target_path, "r") as file:
        result_dict = json.load(file)
    raw_inputs = result_dict["prediction"][target_index - 1]  # output of layer before
    inputs = metadata.convert_json_to_np(raw_inputs)

    input_path = join(target_dir, "original.npy")
    np.save(input_path, inputs)
    print(f"Saved sample at {input_path}")

    rng = np.random.default_rng(1337)

    for i in range(NUM_PERMUTATIONS):
        permutation = rng.permutation(inputs.flatten()).reshape(
            inputs.shape
        )  # permutation uses ONLY 1st axis, so flatten and reshape
        assert not np.all(permutation == inputs)
        current_path = join(target_dir, f"permutation_i{i}.npy")
        np.save(current_path, permutation)

    random_input = rng.random(inputs.shape)
    np.save(join(target_dir, "random.npy"), random_input)


def extract_layer(target_model, target_layer, model_dir):
    # get model and layer
    model = models.get_model(target_model)
    layer = model.get_layer(target_layer)

    # extract layer as is
    extracted_layer = keras.models.Sequential(
        layer, name=f"{target_model}-{target_layer}"
    )
    extracted_layer.build(input_shape=layer.input_shape)
    extracted_layer.save(join(model_dir, f"{extracted_layer.name}.h5"))


def run_experiment(instance: gcloud.GcloudInstanceInfo, result_configs, model_name):
    connection = Connection(instance.ip)

    # download samples and models
    print(f"Downloading model {model_name} on {instance.name}...")
    connection.run(
        f"gsutil -m cp 'gs://forennsic-conv2/input_distribution/{model_name}.h5' $HOME/Projects/forennsic/experiments/data/models",
        hide=True,
    )

    for result_config in result_configs:
        input_name, target_dir = result_config

        input_path = f"/tmp/{input_name}.npy"

        print(f"Downloading sample on {instance.name}...")
        connection.run(
            f"gsutil -m cp 'gs://forennsic-conv2/input_distribution/{input_name}.npy' '{input_path}'",
            hide=True,
        )

        with connection.cd("$HOME/Projects/forennsic/experiments"):
            with connection.prefix("source ../venv/bin/activate"):
                print(f"Predicting on instance {instance.name}...")
                cmd = f"innfcli predict {model_name} '{input_path}' --output_path /tmp/output.json"
                connection.run(cmd, hide=True)

                target_path = join(target_dir, f"{instance.name.replace('-','_')}.json")
                connection.get("/tmp/output.json", target_path)


@click.command()
@click.option("--clean/--no-clean", default=False)
@click.option("--upload/--no-upload", default=False)
@click.option("--stop/--no-stop", default=False)
def main(clean: bool, upload: bool, stop: bool):
    if clean:
        print("Cleaning up previous data in cloud storage")
        invoke.run("gsutil rm -r 'gs://forennsic-conv2/input_distribution/*'")

    if upload:
        print("Extracting target models")
        tempdir = prepare(TARGET_MODEL, TARGET_LAYER, SAMPLE_INDEX)
        print("Uploading models and samples")
        local_sample_path = join(tempdir, "*.npy")
        local_model_path = join(tempdir, "*.h5")
        upload_command = f"gsutil -m cp {local_sample_path} {local_model_path} gs://forennsic-conv2/input_distribution/"
        invoke.run(upload_command)

    instances = gcloud.get_instances()
    for instance in instances:
        remote.update_host_keys(instance.ip)

    input_name = ["original", "random"]
    input_name += [f"permutation_i{i}" for i in range(NUM_PERMUTATIONS)]

    model_name = f"{TARGET_MODEL}-{TARGET_LAYER}"
    result_configs = []
    for input_name in input_name:
        target_dir = join(RESULT_DIR, model_name, input_name)
        os.makedirs(target_dir, exist_ok=True)
        result_configs.append((input_name, target_dir))

    try:
        with parallel_backend("loky"):
            Parallel()(
                delayed(run_experiment)(instance, result_configs, model_name)
                for instance in instances
            )
    finally:
        if stop and len(instances) > 0:
            instance_names = " ".join([instance.name for instance in instances])
            invoke.run(
                f"gcloud compute instances stop {instance_names} --zone europe-west4-a -q"
            )


if __name__ == "__main__":
    main()
