from typing import Dict, List, Tuple
import click
import pandas as pd
import numpy as np
import itertools
import more_itertools
from common import convert_to_tikz_number

OFFSET = 0.03
TRANSITION_OFFSET = 0.05
F_VALUES = [1, 3]
K_VALUES = [2, 3, 4, 5, 6, 7]


def get_offsets(df: pd.DataFrame) -> Dict:
    offset_configurations = {}

    occupancies = df.groupby(["equivalence_classes"]).count()
    for row in occupancies.iterrows():
        equivalence_class = row[0]

        count = row[1]["c"]
        starting_offset = -count / 2 * OFFSET
        offsets = {}

        # get all configurations for current location
        occupants = df.query(f"equivalence_classes == {equivalence_class}").sort_values(
            ["f", "k"]
        )
        for i, occupant in enumerate(occupants.iloc):
            # extract configuration
            config = occupant.name[:2]
            offsets[config] = starting_offset + i * OFFSET

        offset_configurations[equivalence_class] = offsets

    return offset_configurations


@click.group()
def main():
    pass


@main.command()
@click.argument(
    "data_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
def subway(data_path: str):
    df: pd.DataFrame = pd.read_feather(data_path)
    df = df.query(f"f in {F_VALUES} and c==3").sort_values(["k", "f"])
    df["x"] = np.log10(df.multiplications).round()

    df = df.groupby(["f", "k", "i"]).max()

    # get offset
    offset_configurations = get_offsets(df)

    # fs = [1, 3]

    for f, k in itertools.product(F_VALUES, K_VALUES):
        subset = df.query(f"f == {f} and k == {k}")
        if len(subset) == 0:
            continue

        print(
            f"\\begin{{scope}}[f{convert_to_tikz_number(f)},k{convert_to_tikz_number(k)}]"
        )

        print("\\draw")

        for i, row in enumerate(subset.iloc):
            equivalence_classes = row["equivalence_classes"]

            config = (f, k)

            offset = offset_configurations[equivalence_classes][config]
            y = equivalence_classes + offset
            multiplications = np.log10(row["multiplications"])

            if i > 0:
                old_eqs = subset.iloc[i - 1]["equivalence_classes"]
                if old_eqs == equivalence_classes:
                    print("-- ", end="")

                else:
                    old_x = subset.iloc[i - 1]["x"]
                    midway_x = old_x + (row["x"] - old_x) / 2
                    transition_x = (
                        midway_x - k * TRANSITION_OFFSET
                    )  # reusing offset does not work, so generate a new one quickly

                    print(f"-| {transition_x, y} -- ")

            print(f"{multiplications,y} coordinate ({convert_to_tikz_number(i)})")
        print(";")

        print("\\draw[marker]")
        for i, row in enumerate(subset.iloc):
            print(f"({convert_to_tikz_number(i)}) circle (\\circlesize)")
        print(";")

        print("\\end{scope}")


@main.command()
def table():
    df = pd.read_feather(
        "results/num_multiplications/single_layer/2021_11_03_11:21:55/data.feather"
    )
    df = df.query(f"f in {F_VALUES} and c==3").sort_values(["k", "f"])
    df = df.groupby(["f", "k", "i"]).max()
    df = df["equivalence_classes"].reset_index()

    max_eqcs = df.equivalence_classes.max()

    # filter for fs
    f_1 = df.query("f == 1")
    f_3 = df.query("f == 3")
    # chunk into individual ks
    f_1_ks = list(more_itertools.chunked(f_1.equivalence_classes, 5))
    f_3_ks = list(more_itertools.chunked(f_3.equivalence_classes, 5))
    # get ith element from each k over all fs
    rows = list(zip(*f_1_ks, *f_3_ks))

    def convert_eqcs(eqcs):
        if eqcs == max_eqcs:
            return f"\hl{{ {eqcs} }}"
        return str(eqcs)

    for i, row in enumerate(rows):
        print(f"{i+3} & & {'&'.join(map(convert_eqcs, row))} \\\\")


if __name__ == "__main__":
    main()
