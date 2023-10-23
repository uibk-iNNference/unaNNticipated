import random
from typing import List

from tensorflow.python.framework.dtypes import QUANTIZED_DTYPES

random.seed(42)

# Numpy RNG
import numpy as np

np.random.seed(42)

# TF RNG
from tensorflow.python.framework import random_seed

random_seed.set_seed(42)

import itertools
from dataclasses import dataclass
from math import ceil, sqrt
import tempfile
from os.path import join, basename
import os
import invoke
import click
from fabric import Connection
from joblib.parallel import Parallel, delayed, parallel_backend


import numpy as np
from innfrastructure import models, gcloud, remote
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

F = [1, 3, 256]
K = [3, 7, 11]
C = [3]

RESULT_DIR = join("results", "max_resolution")


@dataclass
class RunConfiguration:
    f: int
    k: int
    i: int
    c: int

    def get_name(self) -> str:
        return f"f{self.f}_k{self.k}_i{self.i}_c{self.c}"


@dataclass
class ResultConfiguration:
    model_name: str
    sample_path: str
    target_dir: str


def build_conv_model(run_config: RunConfiguration) -> Sequential:
    f, k, i, c = run_config.f, run_config.k, run_config.i, run_config.c

    return Sequential(
        name=run_config.get_name(),
        layers=[
            # layers.Dense(dim, input_shape=(dim,), activation="relu"),
            layers.Conv2D(
                f,
                k,
                input_shape=(i, i, c),
                padding="same",
            ),
        ],
    )


def run_experiment(
    instance: gcloud.GcloudInstanceInfo, run_config: RunConfiguration, target_dir: str
):
    config_name = run_config.get_name()

    connection = Connection(instance.ip)
    # download samples and models
    print(f"Downloading models on {instance.name}...")
    connection.run(
        f"gsutil cp gs://forennsic-conv2/max_resolution/{config_name}.npy /tmp/",
        hide=True,
    )
    connection.run(
        f"gsutil cp gs://forennsic-conv2/max_resolution/{config_name}.h5 $HOME/Projects/forennsic/experiments/data/models",
        hide=True,
    )

    with connection.cd("$HOME/Projects/forennsic/experiments"):
        with connection.prefix("source ../venv/bin/activate"):
            print(f"Predicting {config_name} on {instance.name}")
            cmd = f"innfcli predict {config_name} '/tmp/{config_name}.npy' --output_path /tmp/output.json"
            connection.run(cmd, hide=True)

            local_name = join(target_dir, f"{instance.name.replace('-','_')}.json")
            connection.get("/tmp/output.json", local_name)


@click.command()
@click.option("-f", type=int, default=512)
@click.option("-k", type=int, default=3)
@click.option("-i", type=int, default=4)
@click.option("-c", type=int, default=128)
@click.option("--upload/--no-upload", default=False)
@click.option("--stop/--no-stop", default=False)
def main(f: int, k: int, i: int, c: int, upload: bool, stop: bool):
    run_config = RunConfiguration(f, k, i, c)

    if upload:
        tempdir = tempfile.mkdtemp(prefix="max_resolution-")

        model = build_conv_model(run_config)
        model_path = join(tempdir, f"{run_config.get_name()}.h5")
        model.save(model_path)

        input_shape = (1,) + model.input_shape[1:]
        rng = np.random.default_rng(1337)
        sample = rng.random(input_shape)
        sample_path = join(tempdir, f"{run_config.get_name()}.npy")
        np.save(sample_path, sample)

        print("Uploading model and sample")
        upload_command = f"gsutil -m cp '{sample_path}' '{model_path}' gs://forennsic-conv2/max_resolution/"
        invoke.run(upload_command)

    target_dir = join(RESULT_DIR, run_config.get_name())
    try:
        os.makedirs(target_dir)
    except FileExistsError:
        os.rmdir(target_dir)  # if it exists, we want it emptied first
        os.makedirs(target_dir)

    instances = gcloud.get_instances()
    for instance in instances:
        remote.update_host_keys(instance.ip)

    try:
        with parallel_backend("loky"):
            Parallel()(
                delayed(run_experiment)(instance, run_config, target_dir)
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
