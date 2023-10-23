from glob import glob
from os.path import join
import numpy as np
from numpy.core.records import array

paths = glob(join("results", "secondary", "debugging", "*.npy"))

for path in paths:
    for other_path in paths:
        if other_path == path:
            continue

        a = np.load(path)
        b = np.load(other_path)

        if np.any(a != b):
            print(f"Paths {path} and {other_path} are not identical")
