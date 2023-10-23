from glob import glob
from os.path import join, basename, splitext
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import click


@click.group()
def main():
    pass


@main.command("samples")
@click.argument("sample_dir", type=click.Path(dir_okay=True, file_okay=False))
def compare_samples(sample_dir: str):
    paths = glob(join(sample_dir, "*.npy"))

    for path in paths:
        conf = splitext(basename(path))[0]

        sample = np.load(path)

        print(f"Parsing {path}...")
        sns.displot(sample.flatten())
        plt.subplots_adjust(top=0.93)
        plt.title(conf)

    plt.show()


@main.command("weights")
@click.argument("weight_dir", type=click.Path(dir_okay=True, file_okay=False))
def compare_weights(weight_dir: str):
    paths = glob(join(weight_dir, "*.h5"))

    from tensorflow.keras.models import load_model

    for path in paths:
        conf = splitext(basename(path))[0]

        m = load_model(path)
        weights, biases = m.get_weights()

        print(f"Parsing {path}...")
        sns.displot(weights.flatten())
        plt.subplots_adjust(top=0.93)
        plt.title(f"{conf} ({np.prod(weights.shape)} total)")

    plt.show()


if __name__ == "__main__":
    main()
