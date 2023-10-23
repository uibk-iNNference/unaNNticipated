import itertools
import os
import tempfile
from datetime import datetime
from os import path
from typing import List

import click
import invoke
import numpy as np
import pandas as pd
from fabric import Connection
from innfrastructure import gcloud, remote
from joblib.parallel import Parallel, delayed, parallel_backend
from tensorflow.keras import layers, models

RESULT_DIR = path.join("results", "dendrogram_consistency")

from dataclasses import dataclass


@dataclass
class RunConfiguration:
    f: int
    k: int
    i: int
    c: int

    def get_name(self) -> str:
        return f"conv_f{self.f}_k{self.k}_i{self.i}x{self.i}x{self.c}"


@dataclass
class ResultConfiguration:
    model_name: str
    sample_path: str
    target_dir: str


# configurations that generated 4 equivalence classes in ablation study
CONFIGURATIONS = [
    RunConfiguration(f=1, k=4, i=103, c=3),
    RunConfiguration(f=1, k=4, i=323, c=3),
    RunConfiguration(f=3, k=6, i=125, c=3),
    RunConfiguration(f=3, k=6, i=393, c=3),
]
NUM_RUNS = 7


def build_conv_models(run_config: RunConfiguration) -> models.Sequential:
    f = run_config.f
    k = run_config.k
    i = run_config.i
    c = run_config.c

    return [
        models.Sequential(
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
        for _ in range(NUM_RUNS)
    ]


def prepare_data(tmpdir: str) -> List[str]:
    rng = np.random.default_rng(1337)
    sample_dir = path.join(tmpdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    model_dir = path.join(tmpdir, "models")
    os.makedirs(model_dir, exist_ok=True)

    for config in CONFIGURATIONS:
        models = build_conv_models(config)

        for i, model in enumerate(models):
            run_name = f"{model.name}_r{i}"
            model_path = path.join(model_dir, f"{run_name}.h5")
            # print(f"Storing the model at {model_path}")
            model.save(model_path)

            for j in range(NUM_RUNS):
                samples = rng.random((1,) + model.input_shape[1:])
                sample_path = path.join(sample_dir, f"{run_name}_s{j}.npy")
                np.save(sample_path, samples)


def run_experiment(
    instance: gcloud.GcloudInstanceInfo, tmpdir: str, configs: List[ResultConfiguration]
):

    connection = Connection(instance.ip)
    # download samples and models
    print(f"Downloading models on {instance.name}...")
    connection.run(
        f"gsutil -m rsync -avz 'gs://forennsic-conv2/{path.basename(tmpdir)}/samples' '/tmp/'",
        hide=True,
    )
    connection.run(
        f"gsutil -m rsync -avz 'gs://forennsic-conv2/{path.basename(tmpdir)}/models' $HOME/Projects/forennsic/experiments/data/models",
        hide=True,
    )

    num_models = len(configs)

    with connection.cd("$HOME/Projects/forennsic/experiments"):
        with connection.prefix("source ../venv/bin/activate"):
            for i, config in enumerate(configs):
                print(f"{instance.name}: {config.model_name} [{i}/{num_models}]")
                cmd = f"innfcli predict {config.model_name} '{config.sample_path}'"
                stdout = connection.run(cmd, hide=True).stdout

                filename = path.join(
                    config.target_dir, f"{instance.name.replace('-','_')}.json"
                )
                with open(filename, "w") as result_file:
                    result_file.write(stdout)


@click.command()
@click.option("--upload/--no-upload", default=False)
def main(upload: bool):
    # tmp = tempfile.mkdtemp(prefix=f"conv_2d_parameters-{timestamp}")
    tmp = path.join(tempfile.tempdir, "dendrogram_consistency")

    os.makedirs(tmp, exist_ok=True)
    result_configs = []

    model_names = [config.get_name() for config in CONFIGURATIONS]

    for name, r, s in itertools.product(model_names, range(NUM_RUNS), range(NUM_RUNS)):
        target_dir = path.join(RESULT_DIR, name, f"r{r}_s{s}")
        os.makedirs(target_dir, exist_ok=True)

        model_name = f"{name}_r{r}"
        sample_path = f"/tmp/{model_name}_s{s}.npy"

        result_configs.append(ResultConfiguration(model_name, sample_path, target_dir))

    if upload:
        prepare_data(tmp)
        print("Uploading models and samples")
        upload_command = f"gsutil -m cp -r '{tmp}' gs://forennsic-conv2/"
        invoke.run(upload_command)

    instances = gcloud.get_instances()
    for instance in instances:
        remote.update_host_keys(instance.ip)

    # for instance in instances:
    #     run_experiment(instance, tmp, result_configs)
    try:
        with parallel_backend("loky"):
            Parallel()(
                delayed(run_experiment)(instance, tmp, result_configs)
                for instance in instances
            )
    finally:
        instance_names = " ".join([instance.name for instance in instances])
        invoke.run(
            f"gcloud compute instances stop {instance_names} --zone europe-west4-a -q"
        )


if __name__ == "__main__":
    main()
