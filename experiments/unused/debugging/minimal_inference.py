# With this file I'm trying to build a small script that produces differences
# It's not supposed to depend on iNNfrastructure, and only use pure tensorflow

from os.path import join, dirname
from os import makedirs
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

samples = np.load(join("data", "datasets", "cifar10", "samples.npy"))
sample = np.expand_dims(samples[0], 0)

with tf.device("/device:CPU:0"):
    minimal_model = load_model(join("data", "models", "cifar10_medium.h5"))
    results = minimal_model(sample)
