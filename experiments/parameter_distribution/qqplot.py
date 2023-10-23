import click
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from innfrastructure import models
from numpy.random import default_rng

TARGET_MODEL = "cifar10_medium"
TARGET_LAYER = "conv2d_11"

a = np.random.normal(5, 5, 250)
b = np.random.rayleigh(5, 250)

percs = np.linspace(0, 100, 21)
qn_a = np.percentile(a, percs)
qn_b = np.percentile(b, percs)


x = np.linspace(np.min((qn_a.min(), qn_b.min())), np.max((qn_a.max(), qn_b.max())))


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
    # drawn = stats.t.rvs(
    #     7.328746001135026, -0.019098451122809123, 0.19420073189894815, size=num_samples
    # )
    drawn = rng.normal(-0.018825512, 0.2262132, size=num_samples)

    percentiles = np.linspace(0, 100, 101)
    q_actual = np.percentile(subsampled, percentiles)
    q_drawn = np.percentile(drawn, percentiles)
    plt.scatter(q_actual, q_drawn)

    plt.xticks(q_actual, percentiles)
    plt.xlabel("percentiles of sample distribution")

    plt.yticks(q_drawn, percentiles)
    plt.ylabel("percentiles of fitted distribution")
    plt.title("Fitted normal distribution")

    plt.yscale("log")
    plt.xscale("log")

    x = np.linspace(
        min(q_actual.min(), q_drawn.min()), max(q_actual.max(), q_drawn.max())
    )
    plt.plot(x, x, ls="--", color="black")

    plt.show()


if __name__ == "__main__":
    main()
