from typing import Dict, List, Tuple
import click
import pandas as pd
import numpy as np
import itertools
from common import convert_to_tikz_number

F_VALUES = [1, 3]
K_VALUES = [2, 3, 4, 5, 6, 7]

Y_RANGE = [0, 3.8]
Y_DIST = Y_RANGE[1] - Y_RANGE[0]

Y_SCALING = 1e5


@click.command()
@click.argument(
    "data_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
def main(data_path: str):
    df: pd.DataFrame = pd.read_feather(data_path)
    df = df.query(f"f in {F_VALUES} and c==3").sort_values(["k", "f"])
    df["x"] = np.log10(df.multiplications)

    df = df.groupby(["f", "k", "i"]).min()
    max_precision = df.precision.max()

    for f, k in itertools.product(F_VALUES, K_VALUES):
        subset = df.query(f"f == {f} and k == {k}")
        if len(subset) == 0:
            continue

        print(
            f"\\begin{{scope}}[f{convert_to_tikz_number(f)},k{convert_to_tikz_number(k)}]"
        )

        print("\\draw")

        for i, row in enumerate(subset.iloc):
            precision = row["precision"]
            scaled_precision = Y_RANGE[0] + precision / max_precision * Y_DIST
            multiplications = np.log10(row["multiplications"])

            if i > 0:
                print("-- ", end="")

            print(
                f"{multiplications,scaled_precision} coordinate ({convert_to_tikz_number(i)})"
            )
        print(";")

        print("\\draw[marker]")
        for i, row in enumerate(subset.iloc):
            print(f"({convert_to_tikz_number(i)}) circle (\\circlesize)")
        print(";")

        print("\\end{scope}")


if __name__ == "__main__":
    main()
