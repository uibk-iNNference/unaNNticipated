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

import invoke
from joblib.parallel import Parallel, parallel_backend, delayed

RESULT_DIR = path.join("results", "num_multiplications", "two_layers")


def build_conv_model(
    f1: int, f2: int, kernel_size: int, input_shape: Tuple[int, int, int]
) -> models.Sequential:
    return models.Sequential(
        name=f"conv_f{f1}x{f2}_k{kernel_size}_i{'x'.join([str(s) for s in input_shape])}",
        layers=[
            # layers.Dense(dim, input_shape=(dim,), activation="relu"),
            layers.Conv2D(f1, kernel_size, input_shape=input_shape, padding="same"),
            layers.Conv2D(f2, kernel_size, padding="same"),
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
        f1 = int(config.f1)
        f2 = int(config.f2)
        k = int(config.k)
        i = int(config.i)
        input_shape = (i, i, f1)
        model = build_conv_model(f1, f2, k, input_shape)

        model_path = path.join(model_dir, f"{model.name}.h5")
        # print(f"Storing the model at {model_path}")
        model.save(model_path)
        names.append(model.name)

        samples = rng.random((1,) + input_shape)
        sample_path = path.join(sample_dir, f"{model.name}.npy")
        np.save(sample_path, samples)

    return names


def run_experiment(instance: gcloud.GcloudInstanceInfo, tmpdir, configs):

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
            for i, (model_name, result_dir) in enumerate(configs.items()):
                print(f"{instance.name}: {model_name} [{i}/{num_models}")
                cmd = f"innfcli predict {model_name} '/tmp/{model_name}.npy'"
                stdout = connection.run(cmd, hide=True).stdout

                filename = path.join(
                    result_dir, f"{instance.name.replace('-','_')}.json"
                )
                with open(filename, "w") as result_file:
                    result_file.write(stdout)


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, readable=True))
@click.option("--upload/--no-upload", default=False)
def main(configuration_path: str, upload: bool):
    configurations = pd.read_csv(configuration_path)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
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
        for instance in gcloud.get_instances():
            gcloud.stop_instance(instance)


if __name__ == "__main__":
    main()
