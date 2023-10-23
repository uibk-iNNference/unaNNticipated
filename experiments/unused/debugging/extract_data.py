import numpy as np
from tensorflow.keras.models import load_model
import sys

samples = np.load("data/datasets/minimal/samples.npy")
sample = samples[0]

print("double sample_values[] = {")
for c in range(sample.shape[-1]):
    print(", ".join(map(str, sample[:, :, c].flatten())), end=",\n")
print("};")

model = load_model("data/models/minimal.h5")

layer = model.layers[-1]
weight = layer.get_weights()[0]
print("double kernel_value[] = {")
# for c in range(weight.shape[2]):
#     for j in range(weight.shape[1]):
#         print(", ".join(map(str, weight[:, j, c, 0].flatten())), end=",\n")
for c in range(weight.shape[-2]):
    print(", ".join(map(str, weight[:, :, c, 0].flatten())), end=",\n")
    # print()
print("};")

print(f"Normal weight:\n{[weight[:,:,c,0] for c in range(2)]}", file=sys.stderr)
print(f"Weight shape: {weight.shape}", file=sys.stderr)
