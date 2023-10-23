import tempfile
import numpy as np
import pandas as pd
import click
from tensorflow.keras import models, layers
from typing import List, Tuple
from datetime import datetime
from os import path
import os
from innfrastructure import gcloud, remote
from fabric import Connection
import json

import invoke
from joblib.parallel import Parallel, parallel_backend, delayed

RESULT_DIR = path.join("results", "num_multiplications", "single_layer")

CONFIGS = [
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


def build_conv_model(
    num_filters: int, kernel_size: int, input_shape: Tuple[int, int, int]
) -> models.Sequential:
    return models.Sequential(
        name=f"conv_f{num_filters}_k{kernel_size}_i{'x'.join([str(s) for s in input_shape])}",
        layers=[
            # layers.Dense(dim, input_shape=(dim,), activation="relu"),
            layers.Conv2D(
                num_filters, kernel_size, input_shape=input_shape, padding="same"
            ),
        ],
    )


def prepare_data(configurations: pd.DataFrame, tmpdir: str) -> List[str]:
    names = []
    rng = np.random.default_rng(1337)
    sample_dir = path.join(tmpdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    model_dir = path.join(tmpdir, "models")
    os.makedirs(model_dir, exist_ok=True)

    for config in configurations.iloc:
        f = int(config.f)
        k = int(config.k)
        c = int(config.c)
        i = int(config.i)
        input_shape = (i, i, c)
        model = build_conv_model(f, k, input_shape)

        model_path = path.join(model_dir, f"{model.name}.h5")
        # print(f"Storing the model at {model_path}")
        model.save(model_path)
        names.append(model.name)

        samples = rng.random((1,) + input_shape)
        sample_path = path.join(sample_dir, f"{model.name}.npy")
        np.save(sample_path, samples)

    return names


def run_experiment(instance: gcloud.GcloudInstanceInfo, tmpdir, configs):
    connection = gcloud.GcloudConnection(instance.ip)
    num_models = len(configs)

    for i, (model_name, result_dir) in enumerate(configs.items()):
        filename = path.join(result_dir, f"{instance.name.replace('-', '_')}.json")
        if os.path.exists(filename):
            continue # don't rerun existing experiments
        connection.docker_run(
            f"/google-cloud-sdk/bin/gsutil -m cp 'gs://forennsic-conv2/conv_2d_parameters/samples/{model_name}.npy' '/tmp/num_multiplications.npy'",
            hide=True,
        )
        connection.docker_run(
            f"/google-cloud-sdk/bin/gsutil -m cp 'gs://forennsic-conv2/conv_2d_parameters/models/{model_name}.h5' "
            "/forennsic/data/models/num_multiplications.h5",
            hide=True,
        )
        print(f"{instance.name}: {model_name} [{i}/{num_models}")
        cmd = f"innfcli predict num_multiplications '/tmp/num_multiplications.npy' --output_path /tmp/output.json"
        connection.docker_run(cmd, hide=True)

        connection.get("/tmp/output.json", filename)
        with open(filename, "r") as test_file:
            print(f"Testing {filename} for validity")
            json.load(test_file)  # do this to fail fast


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, readable=True))
@click.option("--upload/--no-upload", default=False)
def main(configuration_path: str, upload: bool):
    configurations = pd.read_csv(configuration_path)
    timestamp = datetime.now().strftime("%Y_%m_%d")
    # tmp = tempfile.mkdtemp(prefix=f"conv_2d_parameters-{timestamp}")
    tmp = path.join(tempfile.tempdir, "conv_2d_parameters")

    os.makedirs(tmp, exist_ok=True)
    model_names = prepare_data(configurations, tmp)
    base_dir = path.join(RESULT_DIR, timestamp)
    result_configs = {}
    for name in model_names:
        target_dir = path.join(base_dir, name)
        os.makedirs(target_dir, exist_ok=True)
        result_configs[name] = target_dir

    if upload:
        print("Uploading models and samples")
        upload_command = f"gsutil -m cp -r '{tmp}' gs://forennsic-conv2/"
        invoke.run(upload_command)

    gcloud.ensure_configs_running(CONFIGS)
    instances = [
        instance for instance in gcloud.get_instances() if instance.name in CONFIGS
    ]
    assert len(instances) == len(CONFIGS)

    # for instance in instances:
    #     run_experiment(instance, tmp, result_configs)
    with parallel_backend("loky"):
        Parallel()(
            delayed(run_experiment)(instance, tmp, result_configs)
            for instance in instances
        )

    gcloud.cleanup_configs(CONFIGS)


if __name__ == "__main__":
    main()
