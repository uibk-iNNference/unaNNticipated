from glob import glob
from os.path import join, basename, splitext
from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import click
import seaborn as sns


@click.command()
@click.argument("sample_paths", nargs=-1, required=True)
@click.option("--generate-random/--no-generate-random", type=bool, default=False)
def main(sample_paths: List[str], generate_random: bool):
    rows = []

    for path in sample_paths:
        sample = np.load(path)

        if len(sample.shape) == 4:
            sample = sample[0]

        for value in sample.flatten():
            rows.append({"sample": path, "value": value})

    if generate_random:
        rng = np.random.default_rng(1337)
        num_draws = 1024
        random_sample = rng.random(num_draws)

        rows += [{"sample": "random", "value": value} for value in random_sample]

    df = pd.DataFrame(rows)
    sns.displot(data=df, col="sample", x="value")
    plt.show()


if __name__ == "__main__":
    main()
