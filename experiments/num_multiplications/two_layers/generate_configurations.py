import click
from dataclasses import dataclass
import pandas as pd
import itertools


@dataclass
class TwoLayerRunConfig:
    f1: int
    f2: int
    k: int
    i: int

    def get_multiplications(self) -> int:
        return self.k * self.k * self.i * self.i * self.f1 * (self.f1 + self.f2)

    def __str__(self) -> str:
        return f"f1:{self.f1} f2:{self.f2} k:{self.k} i:{self.i}, total size: {self.get_multiplications()}"


@click.command()
@click.argument("target_path", type=click.Path())
def main(target_path: str):
    # generate run configs
    f1s = [1, 2, 3, 4]
    f2s = [1, 2, 3, 4]
    ks = [2, 5, 7]

    configs = {1e3: [], 1e4: [], 1e6: [], 1e7: [], 1e8: []}

    candidates = itertools.product(f1s, f2s, ks)
    for candidate in candidates:
        # filename template is
        f1, f2, k = candidate
        # multiplications:
        # l1: k^2 * i^2 * f1^2
        # l2: k^2 * i^2 * f1 * f2
        # total: k^2 * i^2 * f1 * (f1+f2)
        partial_mults = k * k * f1 * (f1 + f2)

        for size in configs:
            if partial_mults > size:
                continue

            i = 3
            old_distance = abs(size - partial_mults * i * i)
            while True:
                i += 1
                new_distance = abs(size - partial_mults * i * i)

                # distance should be decreasing until it starts to increase again
                if new_distance > old_distance:
                    i -= 1
                    break

                old_distance = new_distance

            candidate_config = TwoLayerRunConfig(f1, f2, k, i)

            if candidate_config.get_multiplications() > size * 10:
                config_before = TwoLayerRunConfig(f1, f2, k, i - 1)
                breakpoint()

            configs[size].append(candidate_config)

    rows = []
    for size, parameter_combinations in configs.items():
        for parameter_combination in parameter_combinations:
            rows.append(
                {
                    "target_size": int(size),
                    "f1": parameter_combination.f1,
                    "f2": parameter_combination.f2,
                    "k": parameter_combination.k,
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
    print(f"\nTotal configurations: {len(df)}")

    if not target_path.endswith(".csv"):
        target_path += ".csv"
    df.to_csv(target_path)

    return df


if __name__ == "__main__":
    main()
