"""This file is the run-once preparation for CPU predictions.
CPU predictions can be used as a main measurement to assign equivalence classes to different CPUs.
This script will train a model and store it for use on different instances.
"""
from typing import Tuple
import tensorflow as tf
import os
from os.path import join
import random
from innfrastructure import config
from glob import glob
import numpy as np
import tempfile
import click
import functools

from tensorflow.python.keras.backend import _CURRENT_SCRATCH_GRAPH

random.seed(42)

# Numpy RNG
import numpy as np

np.random.seed(42)

# TF RNG
from tensorflow.python.framework import random_seed

random_seed.set_seed(42)

import tensorflow_datasets as tfds
from tensorflow.keras import applications, layers, models, callbacks


def preprocess_cifar(dataset):
    def normalize(sample, label):
        return tf.cast(sample, tf.float32) / 255.0, label

    return dataset.map(normalize).cache().shuffle(len(dataset), seed=42).batch(32)


def build_minimal_model() -> models.Sequential:
    return models.Sequential(
        name="minimal_debug",
        layers=[
            # layers.Dense(dim, input_shape=(dim,), activation="relu"),
            layers.Conv2D(2, 3, input_shape=(2, 2, 2), padding="same"),
        ],
    )


def prepare_minimal():
    print("Preparing minimal")
    model = build_minimal_model()
    save_model(model)

    # ensure dataset directory
    data_dir = join("data", "datasets", "minimal")
    os.makedirs(data_dir, exist_ok=True)

    # generate some samples
    samples = np.abs(np.random.random((100,) + model.input_shape[1:]))
    print(f"Saving samples at {data_dir}")
    np.save(join(data_dir, "samples.npy"), samples)

    single_sample = samples[:1]
    np.save(join(data_dir, "single_sample.npy"), single_sample)


def save_model(model):
    target_path = join(config.MODEL_DIR, f"{model.name}.h5")

    try:
        existing_model = models.load_model(target_path)

        for layer, other_layer in zip(model.layers, existing_model.layers):
            for weight, other_weight in zip(
                layer.get_weights(), other_layer.get_weights()
            ):
                np.testing.assert_allclose(weight, other_weight)

    except OSError:
        print("No pre-existing models found")
        model.save(target_path)
    except AssertionError:
        print("New model is different from old")
        _, target_path = tempfile.mkstemp(prefix=model.name, suffix=".h5")

    print(f"Saving new model at {target_path}")
    model.save(target_path)


def load_deep_weeds():
    return tfds.load(
        "deep_weeds", split=["train[:85%]", "train[85%:]"], as_supervised=True
    )  # only has train set, so split


def preprocess_deep_weeds(dataset):
    def _normalize(image, label):
        return tf.cast(
            tf.image.resize(image, [224, 224]), tf.float32
        ) / 255.0, tf.one_hot(label, depth=9)

    return (
        dataset.map(_normalize)
        .cache()
        .shuffle(len(dataset), seed=42)
        .repeat()
        .batch(32)
    )


def prepare_deep_weeds():
    print("Preparing deepweeds (256x256x3)")

    # extract samples
    num_samples = 400
    _, extraction_ds = load_deep_weeds()
    samples = np.empty((num_samples, 224, 224, 3))
    labels = np.empty((num_samples,))

    for i, (sample, label) in enumerate(extraction_ds):
        if i >= num_samples:
            break

        current_sample = (
            tf.image.resize(sample, [224, 224]).numpy().astype(np.float32) / 255.0
        )
        samples[i] = current_sample

        labels[i] = label.numpy()

    data_dir = join("data", "datasets", "deep_weeds")
    os.makedirs(data_dir, exist_ok=True)
    print(f"Saving samples and labels  to {data_dir}")
    np.save(join(data_dir, "samples.npy"), samples)
    np.save(join(data_dir, "labels.npy"), labels)
    return

    ##########################################################
    # THE FOLLOWING IS NOT USED IN THE PAPER
    # INSTEAD WE USE THE AUTHOR'S MODEL
    ##########################################################
    extractor = applications.ResNet50V2(
        include_top=False, weights=None, input_shape=(256, 256, 3)
    )
    flattened = layers.Flatten()(extractor.outputs[0])
    classifier = layers.Dense(9, activation="softmax")(flattened)
    model = models.Model(inputs=extractor.inputs, outputs=classifier, name="deep_weeds")

    (
        train_ds,
        test_ds,
    ) = load_deep_weeds()
    train_ds, test_ds = preprocess_deep_weeds(train_ds), preprocess_deep_weeds(test_ds)

    log_dir = join("logs", model.name)
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        epochs=30,
        steps_per_epoch=80,
        validation_data=test_ds,
        validation_steps=4,
        callbacks=[tensorboard_callback],
    )

    loss, acc = model.evaluate(test_ds, steps=80)
    print(f"Final validation accuracy: {acc}")

    save_model(model)

    print("Done with deep_weeds :)")


def prepare_cifar10():
    print("Preparing cifar10")

    def _prepare_model(size_descriptor, build_func, no_train=False):
        print(f"Preparing {size_descriptor} model")
        model = build_func()
        cifar10_train_model(model, no_train=no_train)
        save_model(model)

    _prepare_model(
        "small_untrained",
        functools.partial(
            cifar10_build_small, "cifar10_small_untrained"
        ),
        no_train=True,
    )
    # _prepare_model(
    #     "cifar10_small",
    #     functools.partial(
    #         cifar10_build_small, "cifar10_small"
    #     ),
    #     no_train=True,
    # )
    # _prepare_model(
    #     "small_sigmoid_untrained",
    #     functools.partial(
    #         cifar10_build_small_sigmoid, "cifar10_small_sigmoid_untrained"
    #     ),
    #     no_train=True,
    # )
    # _prepare_model(
    #     "medium",
    #     functools.partial(cifar10_build_medium, "cifar10_medium"),
    #     no_train=True,
    # )
    # _prepare_model(
    #     "medium_untrained",
    #     functools.partial(cifar10_build_medium, "cifar10_medium_untrained"),
    #     no_train=True,
    # )
    # _prepare_model(
    #     "medium_sigmoid_untrained",
    #     functools.partial(
    #         cifar10_build_medium_sigmoid, "cifar10_medium_sigmoid_untrained"
    #     ),
    #     no_train=True,
    # )
    # _prepare_model(
    #     "medium_sigmoid",
    #     functools.partial(cifar10_build_medium_sigmoid, "cifar10_medium_sigmoid"),
    #     no_train=False,
    # )
    # _prepare_model(
    #     "medium_sigmoid_maxpool",
    #     functools.partial(
    #         cifar10_build_medium_sigmoid_maxpool, "cifar10_medium_sigmoid_maxpool"
    #     ),
    #     no_train=False,
    # )
    # _prepare_model(
    #     "medium_sigmoid_maxpool_untrained",
    #     functools.partial(
    #         cifar10_build_medium_sigmoid_maxpool,
    #         "cifar10_medium_sigmoid_maxpool_untrained",
    #     ),
    #     no_train=True,
    # )
    # _prepare_model("large", cifar10_build_large)

    # ensure cifar directory
    data_dir = join("data", "datasets", "cifar10")
    os.makedirs(data_dir, exist_ok=True)

    samples, labels = cifar10_extract_samples()
    print(f"Saving data and labels to {data_dir}")
    np.save(join(data_dir, "samples.npy"), samples)
    np.save(join(data_dir, "labels.npy"), labels)

    print("Done with cifar10 :)")


def cifar10_extract_samples(num_samples=400):
    (test_ds,) = tfds.load("cifar10", split=["test"], as_supervised=True)
    samples = np.empty((num_samples, 32, 32, 3))
    labels = np.empty((num_samples,))
    dataset_iterator = iter(test_ds)
    for i in range(num_samples):
        sample, label = dataset_iterator.get_next()
        current_sample = sample.numpy() / 255
        samples[i] = current_sample
        labels[i] = label.numpy()

    return samples, labels


def cifar10_build_small(name: str):
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(3, 3, padding="same", kernel_initializer="he_uniform")(inputs)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(5, 5, padding="same", kernel_initializer="he_uniform")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)

    x = layers.Dense(128, kernel_initializer="he_uniform")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dense(128, kernel_initializer="he_uniform")(x)
    x = layers.Activation("relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = models.Model(inputs, outputs, name=name)
    return model


def cifar10_build_small_sigmoid(name: str):
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(3, 3, padding="same", kernel_initializer="glorot_uniform")(inputs)
    x = layers.Activation("sigmoid")(x)
    x = layers.AvgPool2D()(x)
    x = layers.Conv2D(5, 5, padding="same", kernel_initializer="glorot_uniform")(x)
    x = layers.Activation("sigmoid")(x)
    x = layers.AvgPool2D()(x)
    x = layers.Flatten()(x)

    x = layers.Dense(128, kernel_initializer="glorot_uniform")(x)
    x = layers.Activation("sigmoid")(x)
    x = layers.Dense(128, kernel_initializer="glorot_uniform")(x)
    x = layers.Activation("sigmoid")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = models.Model(inputs, outputs, name=name)
    return model


def cifar10_build_medium(name: str):
    # quick reference https://www.researchgate.net/profile/Paolo_Napoletano/publication/322476121/figure/tbl1/AS:668726449946625@1536448218498/ResNet-18-Architecture.png
    inputs = layers.Input(shape=(32, 32, 3))
    block_inputs = inputs
    x = layers.Conv2D(64, 7, strides=(2, 2), padding="same")(block_inputs)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    def res_net_segment(segment_inputs, num_filters, stride=True):
        # each segment consists of two residual blocks
        block_inputs = segment_inputs
        for i in range(2):
            if stride and i == 0:
                strides = (2, 2)
            else:
                strides = (1, 1)

            x = layers.Conv2D(
                num_filters,
                kernel_size=(3, 3),
                strides=strides,
                kernel_initializer="he_uniform",
                padding="same",
            )(block_inputs)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

            x = layers.Conv2D(
                num_filters,
                kernel_size=(3, 3),
                kernel_initializer="he_uniform",
                padding="same",
            )(x)

            if stride and i == 0:
                scaled_block_inputs = layers.Conv2D(
                    num_filters,
                    1,
                    strides=(2, 2),
                    kernel_initializer="ones",
                    use_bias=False,
                )(block_inputs)
            else:
                scaled_block_inputs = block_inputs

            x = layers.Add()([scaled_block_inputs, x])
            x = layers.BatchNormalization()(x)
            block_inputs = layers.ReLU()(x)

        return x

    x = res_net_segment(x, 64, stride=False)
    x = res_net_segment(x, 128)
    x = res_net_segment(x, 256)
    x = res_net_segment(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs, x, name=name)
    return model


# create the cifar10 model but with sigmoid activation, Xavier initialization, and mean pooling
def cifar10_build_medium_sigmoid(name: str):
    # quick reference https://www.researchgate.net/profile/Paolo_Napoletano/publication/322476121/figure/tbl1/AS:668726449946625@1536448218498/ResNet-18-Architecture.png
    inputs = layers.Input(shape=(32, 32, 3))
    block_inputs = inputs
    x = layers.Conv2D(64, 7, strides=(2, 2), padding="same")(block_inputs)
    x = layers.AvgPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    def res_net_segment(segment_inputs, num_filters, stride=True):
        # each segment consists of two residual blocks
        block_inputs = segment_inputs
        for i in range(2):
            if stride and i == 0:
                strides = (2, 2)
            else:
                strides = (1, 1)

            x = layers.Conv2D(
                num_filters,
                kernel_size=(3, 3),
                strides=strides,
                kernel_initializer="glorot_uniform",  # Xavier normal initializer
                padding="same",
            )(block_inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("sigmoid")(x)

            x = layers.Conv2D(
                num_filters,
                kernel_size=(3, 3),
                kernel_initializer="glorot_uniform",  # Xavier normal initializer
                padding="same",
            )(x)

            if stride and i == 0:
                scaled_block_inputs = layers.Conv2D(
                    num_filters,
                    1,
                    strides=(2, 2),
                    kernel_initializer="ones",
                    use_bias=False,
                )(block_inputs)
            else:
                scaled_block_inputs = block_inputs

            x = layers.Add()([scaled_block_inputs, x])
            x = layers.BatchNormalization()(x)
            block_inputs = layers.Activation("sigmoid")(x)

        return x

    x = res_net_segment(x, 64, stride=False)
    x = res_net_segment(x, 128)
    x = res_net_segment(x, 256)
    x = res_net_segment(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs, x, name=name)
    return model


def cifar10_build_medium_sigmoid_maxpool(name: str):
    # quick reference https://www.researchgate.net/profile/Paolo_Napoletano/publication/322476121/figure/tbl1/AS:668726449946625@1536448218498/ResNet-18-Architecture.png
    inputs = layers.Input(shape=(32, 32, 3))
    block_inputs = inputs
    x = layers.Conv2D(64, 7, strides=(2, 2), padding="same")(block_inputs)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    def res_net_segment(segment_inputs, num_filters, stride=True):
        # each segment consists of two residual blocks
        block_inputs = segment_inputs
        for i in range(2):
            if stride and i == 0:
                strides = (2, 2)
            else:
                strides = (1, 1)

            x = layers.Conv2D(
                num_filters,
                kernel_size=(3, 3),
                strides=strides,
                kernel_initializer="glorot_uniform",  # Xavier normal initializer
                padding="same",
            )(block_inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("sigmoid")(x)

            x = layers.Conv2D(
                num_filters,
                kernel_size=(3, 3),
                kernel_initializer="glorot_uniform",  # Xavier normal initializer
                padding="same",
            )(x)

            if stride and i == 0:
                scaled_block_inputs = layers.Conv2D(
                    num_filters,
                    1,
                    strides=(2, 2),
                    kernel_initializer="ones",
                    use_bias=False,
                )(block_inputs)
            else:
                scaled_block_inputs = block_inputs

            x = layers.Add()([scaled_block_inputs, x])
            x = layers.BatchNormalization()(x)
            block_inputs = layers.Activation("sigmoid")(x)

        return x

    x = res_net_segment(x, 64, stride=False)
    x = res_net_segment(x, 128)
    x = res_net_segment(x, 256)
    x = res_net_segment(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs, x, name=name)
    return model


def cifar10_build_large():
    extractor = applications.ResNet50V2(
        include_top=False, weights=None, input_shape=(32, 32, 3)
    )
    flattened = layers.Flatten()(extractor.outputs[0])
    classifier = layers.Dense(10, activation="softmax")(flattened)
    model = models.Model(
        inputs=extractor.inputs, outputs=classifier, name="cifar10_large"
    )
    return model


def cifar10_train_model(model, no_train=False):
    # train model
    (train_ds, test_ds) = tfds.load(
        "cifar10",
        split=["train", "test"],
        as_supervised=True,
    )
    train_ds = preprocess_cifar(train_ds)
    test_ds = preprocess_cifar(test_ds)
    log_dir = join("logs", model.name)
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    if not no_train:
        model.fit(
            train_ds,
            epochs=30,
            validation_data=test_ds,
            callbacks=[tensorboard_callback],
        )

    return model


@click.command()
@click.option("--eval/--no-eval", default=False)
def prepare(eval: bool):
    # ensure that the data directories exist
    print("Creating project local data directories")
    os.makedirs("data", exist_ok=True)
    # ensure the model directory exists
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # prepare_minimal()
    # prepare_deep_weeds()
    prepare_cifar10()

    if eval:
        # evaluate deepweeds
        _, deep_weeds_test_ds = load_deep_weeds()
        deep_weeds_test_ds = preprocess_deep_weeds(deep_weeds_test_ds)
        model = models.load_model("data/models/deep_weeds.h5")
        loss, acc = model.evaluate(deep_weeds_test_ds, steps=20)
        print(f"{model.name} final validation accuracy: {acc}")

        cifar_test_ds = tfds.load(
            "cifar10",
            split="test",
            as_supervised=True,
        )
        cifar_test_ds = preprocess_cifar(cifar_test_ds)

        def _evaluate_cifar(model_path):
            model = models.load_model(model_path)
            loss, acc = model.evaluate(cifar_test_ds)
            print(f"{model.name} final validation accuracy: {acc}")

        _evaluate_cifar("data/models/cifar10_large.h5")
        _evaluate_cifar("data/models/cifar10_medium.h5")
        _evaluate_cifar("data/models/cifar10_medium_sigmoid.h5")
        _evaluate_cifar("data/models/cifar10_medium_sigmoid_untrained.h5")
        _evaluate_cifar("data/models/cifar10_small.h5")
        _evaluate_cifar("data/models/cifar10_small_sigmoid.h5")
        _evaluate_cifar("data/models/cifar10_small_sigmoid_untrained.h5")


if __name__ == "__main__":
    prepare()
