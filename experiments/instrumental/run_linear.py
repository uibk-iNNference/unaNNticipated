import itertools
import logging
import os
from posixpath import basename
import tempfile
from os.path import join
from typing import Dict, Tuple, List

import click
import invoke
import numpy as np
from innfrastructure import gcloud, models, remote
from tensorflow import keras
from joblib.parallel import Parallel, parallel_backend, delayed

from innfrastructure.gcloud import GcloudConnection

MODEL_CONFIGS = {}
CIFAR_SIZES = ["small", "medium"]
# CIFAR_VARIANTS = ["", "_untrained", "_sigmoid", "_sigmoid_untrained"]
CIFAR_VARIANTS = ["_untrained", "_sigmoid", "_sigmoid_untrained"]
for size in CIFAR_SIZES:
    for variant in CIFAR_VARIANTS:
        MODEL_CONFIGS[f"cifar10_{size}{variant}"] = join(
            "data", "datasets", "cifar10", "samples.npy"
        )
# MODEL_CONFIGS["deep_weeds"] = join("data", "datasets", "deep_weeds", "samples.npy")
RESULT_DIR = join("results", "instrumental")

CLOUD_CONFIGS = [
    "intel-sandy-bridge",
    "intel-ivy-bridge",
    "intel-haswell",
    "intel-broadwell",
    "intel-skylake",
    "intel-ice-lake",
    "intel-cascade-lake",
    "amd-rome",
    "amd-milan",
]


def get_cleaned_name(host, hostname):
    cleaned_host = host.replace("-", "_")
    if hostname is not None:
        cleaned_host = hostname.replace("-", "_")
    return cleaned_host

def prepare(
    model_dir: str, sample_dir: str, target_model: str, sample_path: str
) -> str:
    m = models.get_model(target_model)

    for layer in m.layers:
        if 'activation' not in layer.name:
            continue
        layer.activation = keras.activations.linear


    new_name = m.name.replace("sigmoid", "linear") + "_instrumental"
    linear_model = keras.models.Model(
            m.inputs[0],
            [layer.output for layer in m.layers],
            name=new_name,
    )

    # sanity check new model
    for layer in linear_model.layers:
        assert 'activation' not in layer.name or layer.activation == keras.activations.linear
    assert linear_model.layers[-1].activation == keras.activations.softmax

    linear_model.save(join(model_dir, f"{linear_model.name}.h5"))

    # prepare inputs
    inputs = np.load(sample_path)
    for i in range(3):
        current_inputs = inputs[i : i + 1]
        target_path = join(sample_dir, f"{linear_model.name}-i{i}.npy")
        np.save(target_path, current_inputs)

def run_experiment(
    instance: gcloud.GcloudInstanceInfo, result_configs: List[Tuple[str, str, str]]
):
    connection = GcloudConnection(instance.ip, setup=True)
    print(f"Downloading samples on {instance.name}...")
    sample_target = "instrumental/samples"
    connection.run(f"mkdir -p /tmp/{sample_target}")
    output_dir = "instrumental/output"
    connection.run(f"mkdir -p /tmp/{output_dir}")

    connection.docker_run(
        f"/google-cloud-sdk/bin/gsutil -m cp "
        f"'gs://forennsic-conv2/instrumental/samples/*.npy' '/tmp/{sample_target}'",
        hide=True,
    )
    print(f"Downloading models on {instance.name}...")
    connection.docker_run(
        f"/google-cloud-sdk/bin/gsutil -m cp 'gs://forennsic-conv2/instrumental/models/*.h5' "
        f"/forennsic/data/models",
        hide=True,
    )

    for result_config in result_configs:
        model, target_dir, sample_path = result_config

        sample_config = basename(sample_path).split(".")[0]
        target_path = f"{output_dir}/{sample_config}.json"
        container_target_path = join("/tmp", target_path)
        outer_target_path = join("/tmp", target_path)

        print(f"Predicting {basename(target_dir)} on instance {instance.name}...")
        cmd = f"innfcli predict {model} '{sample_path}' --output_path {container_target_path}"
        connection.docker_run(cmd, hide=True)

        print("Downloading result...")
        local_target_path = join(target_dir, f"{instance.name.replace('-', '_')}.json")
        connection.get(outer_target_path, local_target_path)


@click.command()
@click.option("--clean/--no-clean", default=False)
@click.option("--upload/--no-upload", default=False)
@click.option("--stop/--no-stop", default=False)
def main(clean: bool, upload: bool, stop: bool):
    if clean:
        print("Cleaning up previous data in cloud storage")
        invoke.run("gsutil rm -r 'gs://forennsic-conv2/instrumental/*'")

    if upload:
        print(f"Building instrumental models and inputs for {MODEL_CONFIGS.keys()}")

        # prepare directories
        tempdir = tempfile.mkdtemp(prefix="instrumental_")
        model_dir = join(tempdir, "models")
        os.makedirs(model_dir, exist_ok=True)
        sample_dir = join(tempdir, "samples")
        os.makedirs(sample_dir, exist_ok=True)

        for model, sample_path in MODEL_CONFIGS.items():
            prepare(model_dir, sample_dir, model, sample_path)

        sample_upload_command = f"gsutil -m cp {sample_dir}/*.npy gs://forennsic-conv2/instrumental/samples/"
        invoke.run(sample_upload_command)

        model_upload_command = (
            f"gsutil -m cp {model_dir}/*.h5 gs://forennsic-conv2/instrumental/models/"
        )
        invoke.run(model_upload_command)

    result_configs = []
    for model, i in itertools.product(MODEL_CONFIGS, range(3)):
        instrumental_name = f"{model.replace('sigmoid','linear')}_instrumental"
        target_dir = join(RESULT_DIR, f"{instrumental_name}-i{i}")
        os.makedirs(target_dir, exist_ok=True)
        sample_path = f"/tmp/instrumental/samples/{instrumental_name}-i{i}.npy"

        result_configs.append((instrumental_name, target_dir, sample_path))

    gcloud.ensure_configs_running(CLOUD_CONFIGS)
    instances = [instance for instance in gcloud.get_instances() if instance.name in CLOUD_CONFIGS]

    for instance in instances:
        remote.update_host_keys(instance.ip)

    try:
        # for instance in instances:
        #     run_experiment(instance, result_configs)
        with parallel_backend("loky"):
            Parallel()(
                delayed(run_experiment)(instance, result_configs)
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
