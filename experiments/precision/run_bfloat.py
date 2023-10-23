import itertools
import os
import sys
import tempfile
from copy import deepcopy
from typing import Dict

import numpy as np

from innfrastructure import models, gcloud
import tensorflow as tf
from posixpath import join
from invoke import run

from joblib import parallel_backend, Parallel, delayed

GS_PATH = "gs://forennsic-conv2/precision"
RESULT_DIR = join("results", "precision")
TARGET_CONFIGS = [
    "intel-sandy-bridge",
    "intel-ivy-bridge",
    "intel-haswell",
    "intel-broadwell-2",
    "intel-broadwell-4",
    "intel-skylake",
    "intel-ice-lake",
    "intel-cascade-lake",
    "amd-rome-2",
    "amd-rome-4",
    "amd-rome-8",
    "amd-rome-16",
    "amd-rome-48",
    "amd-milan",
]

TARGET_MODELS = ["cifar10_medium", "cifar10_small", "deep_weeds"]


# load model (cifar10_medium is probably best)
# convert model and inputs to float32 and float64
def cast_model_to_target_types(
    model_name: str = "cifar10_medium",
) -> Dict[str, tf.keras.models.Model]:
    model = models.get_model(model_name)
    config = model.get_config()
    return_models = {}
    for target_type in TARGET_TYPES:
        new_config = deepcopy(config)
        new_config["name"] = generate_typed_name(model_name, target_type)
        type_name = target_type.name
        for layer in new_config["layers"]:
            layer["config"]["dtype"] = type_name

        new_model = tf.keras.models.Model.from_config(new_config)
        for new_layer, old_layer in zip(new_model.layers, model.layers):
            new_layer.set_weights(old_layer.get_weights())

        return_models[type_name] = new_model

    return return_models


def generate_typed_name(model_name, target_type):
    return model_name + f"_{target_type.name}"


# upload to gcs
def upload_data(model_name: str):
    converted_models = cast_model_to_target_types(model_name)
    if "cifar" in model_name:
        sample = np.load(join("data", "datasets", "cifar10", "samples.npy"))[:1]
    elif "deep_weeds" in model_name:
        sample = np.load(join("data", "datasets", "deep_weeds", "samples.npy"))[:1]
    else:
        raise ValueError(f"Unknown model name {model_name}")

    tempdir = tempfile.mkdtemp(prefix="forennsic_precision")
    for target_type, model in converted_models.items():
        print(model.name)
        converted_sample = sample.astype(target_type)
        sample_path = join(tempdir, model.name)
        np.save(sample_path, converted_sample)

        model_path = join(tempdir, model.name + ".h5")
        model.save(model_path)

    run(f"gsutil -m cp {join(tempdir, '*')} {GS_PATH}")


# predict, get results
def run_experiment(instance: gcloud.GcloudInstanceInfo):
    with gcloud.GcloudConnection(instance.ip) as connection:
        connection.docker_run(f"/google-cloud-sdk/bin/gsutil -m cp {GS_PATH}/* /tmp/")
        connection.docker_run("mv /tmp/*.h5 /forennsic/data/models/")

        for model, target_type in itertools.product(TARGET_MODELS, TARGET_TYPES):
            target_model_name = generate_typed_name(model, target_type)
            target_dir = join(RESULT_DIR, target_model_name)
            os.makedirs(target_dir, exist_ok=True)

            result = connection.docker_run(
                f"innfcli predict {target_model_name} /tmp/{target_model_name}.npy",
                hide=True,
            ).stdout
            target_filename = f"{target_model_name}-{instance.name}.json"

            target_path = join(target_dir, target_filename)
            with open(target_path, "w") as f:
                f.write(result)


# compare remaining_precision, eqcs (dendrogram) -> eval.py


def main(upload):
    for type_name in [t.name for t in TARGET_TYPES]:
        subdir = join(RESULT_DIR, type_name)
        os.makedirs(subdir, exist_ok=True)

    if upload:
        for model_name in TARGET_MODELS:
            upload_data(model_name)

    gcloud.ensure_configs_running(TARGET_CONFIGS)
    instances = gcloud.get_instances()

    # for instance in instances:
    #     run_experiment(instance)
    with parallel_backend("loky"):
        Parallel()(delayed(run_experiment)(instance) for instance in instances)


if __name__ == "__main__":
    main(upload="-u" in sys.argv or "--upload" in sys.argv)
