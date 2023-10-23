import click
from dataclasses import dataclass
import pandas as pd


@dataclass
class RunConfig:
    f: int
    k: int
    c: int
    i: int

    def get_multiplications(self) -> int:
        return self.f * self.k * self.k * self.c * self.i * self.i

    def __str__(self) -> str:
        return f"f:{self.f} k:{self.k} c:{self.c} i:{self.i}, total size: {self.get_multiplications()}"


@click.command()
@click.argument("target_path", type=click.Path())
def main(target_path: str):
    # generate run configs
    fs = [1, 2, 3, 4]
    cs = [3]
    ks = [2, 3, 4, 5, 6, 7]

    configs = {1e3: [], 1e4: [], 1e6: [], 1e7: [], 1e8: [], 5e8: []}

    for f in fs:
        for k in ks:
            for c in cs:
                partial_mults = f * k * k * c

                for size in configs:
                    remaining = size / partial_mults
                    i = 3
                    while i * i < remaining / 2:
                        i += 1

                    candidate_config = RunConfig(f, k, c, i)
                    if candidate_config.get_multiplications() > size * 10:
                        continue

                    configs[size].append(candidate_config)

    rows = []
    for size, parameter_combinations in configs.items():
        for parameter_combination in parameter_combinations:
            rows.append(
                {
                    "target_size": int(size),
                    "f": parameter_combination.f,
                    "k": parameter_combination.k,
                    "c": parameter_combination.c,
                    "i": parameter_combination.i,
                    "total_multiplications": parameter_combination.get_multiplications(),
                }
            )
    df = pd.DataFrame(rows)
    df["configuration_mismatch"] = (
        df.total_multiplications - df.target_size
    ).abs() / df.target_size

    print(
        "Mismatch of configurations and target: (mult_actual - mult_target)/mult_target"
    )
    print(df["configuration_mismatch"].describe())
    print()
    print("Configuration with maximum relative mismatch")
    print(df.iloc[df["configuration_mismatch"].idxmax()])

    if not target_path.endswith(".csv"):
        target_path += ".csv"
    df.to_csv(target_path)

    return df


if __name__ == "__main__":
    main()
