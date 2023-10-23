#!/usr/bin/env python3
from os.path import join
import tensorflow as tf
import numpy as np

DEVICE = "/device:GPU:0"

@tf.function(experimental_compile=True)
def recompiled_on_launch(a, b):
  return a + b

recompiled_on_launch(tf.ones([1, 10]), tf.ones([1, 10]))
recompiled_on_launch(tf.ones([1, 100]), tf.ones([1, 100]))
