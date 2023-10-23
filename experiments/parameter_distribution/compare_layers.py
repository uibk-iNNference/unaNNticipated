import os
from innfrastructure import models
import numpy as np
from os.path import join
from numpy.lib.histograms import histogram
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

NUM_BINS = 100
RESULT_DIR = join("results", "parameter_distribution")


def generate_rows(layer, global_min, global_max, i, parameters):
    hist, bin_borders = np.histogram(
        parameters, bins=NUM_BINS, range=(global_min, global_max)
    )
    # normalize histograms
    hist = hist.astype(np.float64) / hist.sum()

    conv_index = int(layer.name.split("_")[1])
    block = (conv_index - 1) // 4
    if conv_index == 1:
        block = -1

    block = int(block)

    within_block = (conv_index - 1) % 4

    return [
        {
            "layer": conv_index,
            "layer_name": layer.name,
            "block": block,
            "within_block": within_block,
            "bin_lower": bin_borders[j],
            "bin_upper": bin_borders[j + 1],
            "count": v,
        }
        for j, v in enumerate(hist)
    ]


def analyze_data(target_model: str):
    model = models.get_model(target_model)
    conv_layers = [layer for layer in model.layers if "conv" in layer.name]
    # get overall bounds
    kernel_mins, kernel_maxs = [], []
    bias_mins, bias_maxs = [], []
    for layer in conv_layers:
        weights = layer.get_weights()

        kernel_mins.append(weights[0].min())
        kernel_maxs.append(weights[0].max())

        try:
            bias_mins.append(weights[1].min())
            bias_maxs.append(weights[1].max())
        except IndexError:
            print(f"Layer {layer.name} has no biases")

    kernel_min = min(kernel_mins)
    kernel_max = max(kernel_maxs)
    bias_min = min(bias_mins)
    bias_max = max(bias_maxs)

    # calculate histograms
    kernel_rows = []
    bias_rows = []

    for i, layer in enumerate(conv_layers):
        print(f"Analyzing layer {i+1}/{len(conv_layers)}")
        kernel = layer.get_weights()[0]
        kernel_rows += generate_rows(layer, kernel_min, kernel_max, i, kernel)

        try:
            bias = layer.get_weights()[1]
            bias_rows += generate_rows(layer, bias_min, bias_max, i, bias)
        except IndexError:
            pass

    # save histograms
    target_dir = join(RESULT_DIR, target_model)
    os.makedirs(target_dir, exist_ok=True)

    kernel_df = pd.DataFrame(kernel_rows)
    # kernel_df.to_feather(join(target_dir, "kernels.feather"))

    bias_df = pd.DataFrame(bias_rows)
    # bias_df.to_feather(join(target_dir, "biases.feather"))

    return kernel_df, bias_df


def main():
    target_model = "cifar10_medium"
    target_dir = join(RESULT_DIR, target_model)

    try:
        kernels = pd.read_feather(join(target_dir, "kernels.feather"))
        biases = pd.read_feather(join(target_dir, "biases.feather"))
    except FileNotFoundError:
        kernels, biases = analyze_data(target_model)

    # kernels = kernels.query("layer_name=='conv2d_13'  )

    print("Plotting data...")
    sns.relplot(
        data=kernels,
        x="bin_lower",
        y="count",
        hue="layer_name",
        style="within_block",
        col="block",
        kind="line",
    )
    plt.title("(Normalized) histograms of kernel parameters")
    plt.subplots_adjust(top=0.95, bottom=0.1)

    sns.relplot(data=biases, x="bin_lower", y="count", hue="layer_name", kind="line")
    plt.suptitle("(Normalized) histograms of bias parameters")
    plt.subplots_adjust(top=0.95, bottom=0.1)
    plt.show()


if __name__ == "__main__":
    main()
