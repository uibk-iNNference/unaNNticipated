import itertools
from posixpath import basename, splitext
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
from os.path import join
import invoke
from fabric import Connection
import tempfile
import click
import logging
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import List

from tensorflow import keras
from joblib.parallel import Parallel, parallel_backend, delayed

from innfrastructure import models, metadata, gcloud, remote

TARGET_MODEL = "cifar10_medium"
TARGET_LAYER = "conv2d_11"
RESULT_DIR = join("results", "parameter_distribution")
NUM_CLASSES = 10
SEEDS = [1337, 42, 69, 420]


def _preprocess(dataset):
    def normalize(sample, label):
        return tf.cast(sample, tf.float32) / 255.0, label

    return dataset.map(normalize).cache().shuffle(len(dataset), seed=42)


def get_samples_per_class(samples_per_class: int = 4):
    dataset = tfds.load("cifar10", split="test", as_supervised=True)
    dataset = _preprocess(dataset)

    samples = [[] for _ in range(NUM_CLASSES)]
    labels = []
    for sample, label in dataset:
        label = label.numpy()
        labels.append(label)
        if len(samples[label]) < samples_per_class:
            samples[label].append(sample.numpy())

        if all([len(s) >= samples_per_class for s in samples]):
            break

    for i in range(NUM_CLASSES):
        assert len(samples[i]) == samples_per_class

    actual_samples = np.vstack(samples)
    return actual_samples


def prepare(target_model: str, target_layer: str) -> str:
    logging.info(f"Extracting layer {target_layer} from {target_model}")

    # prepare target
    tempdir = tempfile.mkdtemp(prefix="parameter_distribution_")

    # get model and layer
    model = models.get_model(target_model)
    layer = model.get_layer(target_layer)

    original_samples = get_samples_per_class()
    input_generator = keras.models.Model(inputs=model.input, outputs=layer.input)
    assert input_generator.input_shape == model.input_shape
    assert input_generator.output_shape == layer.input_shape

    original_inputs = input_generator(original_samples)
    original_filename = "samples_original.npy"
    np.save(join(tempdir, original_filename), original_inputs)

    sample_filenames = extract_samples(original_inputs, SEEDS, tempdir)
    sample_filenames.append(original_filename)

    model_names = extract_models(layer, original_inputs, tempdir)

    return tempdir, model_names, sample_filenames


def extract_samples(original_inputs: np.ndarray, seeds: List[int], target_dir: str):
    target_shape = (NUM_CLASSES,) + original_inputs.shape[1:]

    file_names = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        normal_samples = rng.normal(-0.0188, 0.226, size=target_shape)
        normal_filename = f"samples_normal_seed{seed}.npy"
        np.save(join(target_dir, normal_filename), normal_samples)
        file_names.append(normal_filename)

        rng = np.random.default_rng(seed)
        uniform_samples = rng.random(size=target_shape)
        uniform_filename = f"samples_uniform_seed{seed}.npy"
        np.save(join(target_dir, uniform_filename), uniform_samples)
        file_names.append(uniform_filename)

    return file_names


def _save_model(model_dir: str, model: keras.models.Model):
    model.save(join(model_dir, f"{model.name}.h5"))


def extract_models(layer: keras.layers.Conv2D, inputs: np.array, target_dir: str):
    # extract layer as is
    extracted_layer = keras.models.Sequential(layer, name="extracted_layer")
    extracted_layer.build(input_shape=layer.input_shape)
    assert inputs.shape[1:] == extracted_layer.input_shape[1:]
    _save_model(target_dir, extracted_layer)

    # remove bias
    extracted_config = extracted_layer.get_config()
    extracted_config["name"] = "without_bias"
    without_bias = keras.models.Sequential.from_config(extracted_config)
    kernels, biases = layer.get_weights()
    without_bias.set_weights([kernels, np.zeros_like(biases)])
    without_bias.build()
    assert inputs.shape[1:] == without_bias.input_shape[1:]
    _save_model(target_dir, without_bias)

    # generate equivalent configuration with default distribution
    extracted_config["name"] = "default_distribution"
    default_distribution = keras.models.Sequential.from_config(extracted_config)
    default_distribution.build()
    assert inputs.shape[1:] == default_distribution.input_shape[1:]
    _save_model(target_dir, default_distribution)

    # generate equivalent configuration from fitted distribution
    extracted_config["name"] = "fitted_distribution"
    fitted_distribution = keras.models.Sequential.from_config(extracted_config)
    rng = np.random.default_rng(1337)
    fitted_kernels = rng.normal(-0.018825512, 0.2262132, size=kernels.shape)
    fitted_distribution.set_weights([fitted_kernels, np.zeros_like(biases)])
    fitted_distribution.build()
    assert inputs.shape[1:] == fitted_distribution.input_shape[1:]
    _save_model(target_dir, fitted_distribution)

    return [
        extracted_layer.name,
        without_bias.name,
        default_distribution.name,
        fitted_distribution.name,
    ]


def run_experiment(instance: gcloud.GcloudInstanceInfo, result_configs):
    connection = Connection(instance.ip)
    # download samples and models
    print(f"Downloading samples on {instance.name}...")
    connection.run(
        f"gsutil -m cp 'gs://forennsic-conv2/parameter_distribution/*.npy' '/tmp/'",
        hide=True,
    )
    print(f"Downloading models on {instance.name}...")
    connection.run(
        f"gsutil -m cp 'gs://forennsic-conv2/parameter_distribution/*.h5' $HOME/Projects/forennsic/experiments/data/models",
        hide=True,
    )

    for result_config in result_configs:
        model, sample_filename, target_dir = result_config

        with connection.cd("$HOME/Projects/forennsic/experiments"):
            with connection.prefix("source ../venv/bin/activate"):
                print(
                    f"Predicting {model} and {sample_filename} on instance {instance.name}..."
                )
                cmd = f"innfcli predict {model} '/tmp/{sample_filename}.npy' -b --output_path /tmp/output.json"
                connection.run(cmd, hide=True)

                target_path = join(
                    target_dir,
                    f"{instance.name.replace('-','_')}.json",
                )
                connection.get("/tmp/output.json", target_path)


@click.command()
@click.option("--clean/--no-clean", default=False)
@click.option("--upload/--no-upload", default=False)
@click.option("--stop/--no-stop", default=False)
def main(clean: bool, upload: bool, stop: bool):
    if clean:
        print("Cleaning up previous data in cloud storage")
        invoke.run("gsutil rm -r 'gs://forennsic-conv2/parameter_distribution/*'")

    print("Extracting target models")
    tempdir, model_names, sample_filenames = prepare(TARGET_MODEL, TARGET_LAYER)
    if upload:
        print("Uploading models and samples")
        local_sample_path = join(tempdir, "*.npy")
        local_model_path = join(tempdir, "*.h5")
        upload_command = f"gsutil -m cp {local_sample_path} {local_model_path} gs://forennsic-conv2/parameter_distribution/"
        invoke.run(upload_command)

    # prepare target dirs
    base_dir = join(RESULT_DIR, f"{TARGET_MODEL}-{TARGET_LAYER}")
    result_configs = []
    for model_name, sample_filename in itertools.product(model_names, sample_filenames):
        sample_basename = splitext(basename(sample_filename))[0]
        config = f"{model_name}-{sample_basename}"
        target_dir = join(base_dir, config)
        os.makedirs(target_dir, exist_ok=True)

        result_configs.append((model_name, sample_basename, target_dir))

    instances = gcloud.get_instances()
    for instance in instances:
        remote.update_host_keys(instance.ip)

    try:
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
