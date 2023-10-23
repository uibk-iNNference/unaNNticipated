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


def generate_quantiles(samples, xs, type):
    rows = []
    for x in xs:
        rows += [
            {
                "x": x,
                "quantile": np.count_nonzero(samples < -x)
                / np.count_nonzero(samples < 0),
                "type": type,
                "side": "left",
            },
            {
                "x": x,
                "quantile": np.count_nonzero(samples >= x)
                / np.count_nonzero(samples >= 0),
                "type": type,
                "side": "right",
            },
        ]
    return rows


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

    upper = subsampled.max()
    xs = np.linspace(0, upper, NUM_X_POINTS)

    rows = generate_quantiles(subsampled, xs, "observed")

    # normal
    normal_params = stats.norm.fit(subsampled)
    normal_samples = rng.normal(normal_params[0], normal_params[1], num_samples)
    print(f"Normal params: {normal_params}")
    rows += generate_quantiles(normal_samples, xs, "normal")

    # laplace
    laplace_params = stats.laplace.fit(subsampled)
    laplace_samples = rng.laplace(laplace_params[0], laplace_params[1], num_samples)
    print(f"laplace params: {laplace_params}")
    rows += generate_quantiles(laplace_samples, xs, "laplace")

    # student T
    t_params = stats.t.fit(subsampled)
    t_samples = stats.t.rvs(
        t_params[0], t_params[1], t_params[2], size=num_samples
    )  # TODO: currently not seeded
    print(f"Student t params: {t_params}")
    rows += generate_quantiles(t_samples, xs, "t")

    df = pd.DataFrame(rows)

    # plot
    sns.relplot(data=df, x="x", y="quantile", hue="side", style="type", kind="line")

    # configure plot
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("P(|k_i| > x)")
    plt.title("Candidate distributions")
    plt.subplots_adjust(left=0.12, bottom=0.11, top=0.94)

    # plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
