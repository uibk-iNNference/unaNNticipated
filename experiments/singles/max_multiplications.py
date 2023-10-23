from innfrastructure import models
from tensorflow.keras import layers
import numpy as np

names = ['cifar10_small','cifar10_medium','deep_weeds']
for name in names:
    m = models.get_model(name)
    conv_layers = [layer for layer in m.layers if isinstance(layer, layers.Conv2D)]
    multiplications = [np.prod(l.output_shape[1:-1] + l.get_weights()[0].shape) for l in conv_layers]
    print(f"{name}: {max(multiplications)}")
