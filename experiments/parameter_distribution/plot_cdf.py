from innfrastructure import models
from scipy import stats
import click
from numpy.random import default_rng
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

TARGET_MODEL = "cifar10_medium"
TARGET_LAYER = "conv2d_11"
NUM_X_POINTS = 50


def get_first_over(x, arr):
    """assumes arr is sorted ascending"""
    for i, val in enumerate(arr):
        if val > x:
            return len(arr) - i

    return 0


@click.command()
@click.option("--bias/--no_bias", default=False)
@click.option("--num_samples", default=20000)
def main(bias: bool, num_samples: int):
    model = models.get_model(TARGET_MODEL)
    layer = model.get_layer(TARGET_LAYER)

    kernels, biases = layer.get_weights()

    if bias:
        target_parameters = biases.flatten()
    else:
        target_parameters = kernels.flatten()

    rng = default_rng(1337)
    subsampled = rng.choice(target_parameters, num_samples)

    # upper = subsampled.max()
    xs = np.linspace(subsampled.min(), subsampled.max(), NUM_X_POINTS)

    # clean up actual samples
    left, right = (
        np.abs(subsampled[subsampled < 0]),
        np.abs(subsampled[subsampled >= 0]),
    )
    mirrored_left = np.concatenate((-left, left))
    mirrored_right = np.concatenate((-right, right))

    rows = []
    for x in xs:
        rows += [
            {
                "x": x,
                "y": np.count_nonzero(mirrored_left <= x) / len(mirrored_left),
                "type": "actual",
                "side": "left",
            },
            {
                "x": x,
                "y": np.count_nonzero(mirrored_right <= x) / len(mirrored_right),
                "type": "actual",
                "side": "right",
            },
            {
                "x": x,
                "y": np.count_nonzero(subsampled <= x) / num_samples,
                "type": "actual",
                "side": "both",
            },
        ]

    def fit_distribution(observed, distribution, draw_function, type, side):
        parameters = distribution.fit(observed)
        drawn = draw_function(parameters[0], parameters[1], num_samples)

        print(f"{type} {side} parameters: {parameters}")

        rows = []
        for x in xs:
            rows.append(
                {
                    "x": x,
                    "y": np.count_nonzero(drawn <= x) / len(drawn),
                    "type": type,
                    "side": side,
                },
            )
        return rows

    rows += fit_distribution(mirrored_left, stats.norm, rng.normal, "normal", "left")
    rows += fit_distribution(mirrored_right, stats.norm, rng.normal, "normal", "right")
    rows += fit_distribution(subsampled, stats.norm, rng.normal, "normal", "both")
    rows += fit_distribution(
        mirrored_left, stats.laplace, rng.laplace, "laplace", "left"
    )
    rows += fit_distribution(
        mirrored_right, stats.laplace, rng.laplace, "laplace", "right"
    )
    rows += fit_distribution(subsampled, stats.laplace, rng.laplace, "laplace", "both")
    df = pd.DataFrame(rows)

    # plot
    sns.relplot(data=df, x="x", y="y", hue="side", style="type", kind="line")

    # configure plot
    # plt.yscale("log")
    # plt.xscale("log")
    plt.ylabel("cumulative probability")
    plt.subplots_adjust(left=0.12, bottom=0.11)

    # plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
