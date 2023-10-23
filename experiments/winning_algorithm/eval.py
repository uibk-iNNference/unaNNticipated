import itertools
import functools
import json
import os
import sys
from enum import Enum, auto
from glob import glob
from posixpath import basename, join, splitext
from typing import Dict, List, Set, Union
import numpy as np

import click
import pandas as pd
from innfrastructure import compare, metadata
from more_itertools import flatten
import seaborn as sns
import matplotlib.pyplot as plt

RESULT_DIR = join("results", "winning_algorithm")


class DivergenceReason(Enum):
    INPUT = auto()
    ALGORITHM = auto()
    UNKNOWN = auto()


@click.group()
def main():
    pass


@main.command()
@click.argument("glob_exp", default="")
def get_diverging_algorithms(glob_exp: str):
    paths = get_paths(glob_exp)
    results = get_results(paths)
    # for card, algs in get_algs_per_card(results).items():
    #     print(f"{card}: {', '.join(sorted(algs))}")

    unique_alg_strings = []
    unique_predictions = []
    rows = []

    for result in results:
        all_predictions = result["all_predictions"]
        all_winning_algorithms = result["winning_algorithms"]
        assert len(all_predictions) == len(all_winning_algorithms)

        model_size = result["model_type"].split("_")[1]
        sample_index = result["sample_index"]
        device_name = metadata.get_clean_device_name(result)

        for run_predictions, winning_algorithms in zip(
            all_predictions, all_winning_algorithms
        ):
            algorithm_string = "-".join(winning_algorithms)
            alg_eqc = compare.get_equivalence_class(
                unique_alg_strings, algorithm_string
            )

            final_prediction = run_predictions[-1]
            eqc = compare.get_equivalence_class(
                unique_predictions, final_prediction["output"]["bytes"]
            )

            rows.append(
                {
                    "model_size": model_size,
                    "sample_index": sample_index,
                    "device_name": device_name,
                    "alg_eqc": alg_eqc,
                    "algorithm_string": algorithm_string,
                    "eqc": eqc,
                }
            )

    df = pd.DataFrame(rows)
    df = df.set_index(["model_size", "sample_index", "device_name"]).drop_duplicates()
    groupby = df.groupby(["model_size", "sample_index", "algorithm_string"])
    non_unique = groupby.eqc.nunique() > 1
    non_unique = non_unique[non_unique]
    for idx in non_unique.index:
        matching = df.query(
            f"model_size == '{idx[0]}' & sample_index == {idx[1]} & algorithm_string == '{idx[2]}'"
        )
        print(matching)


@main.command()
@click.argument("glob_exp", default="")
def generate_divergence_traces(glob_exp: str):
    paths = get_paths(glob_exp)
    results = get_results(paths)
    # combine runs from results file
    for result in results:
        sanity_check(result)
        generate_divergence_traces(result)


def get_results(paths):
    results = [load_file(path) for path in paths]
    results = [analyze_result(result) for result in results]
    for result, path in zip(results, paths):
        result["path"] = path

    #     assert_global_alg_determinism(results)
    return results


ROMAN_NUMERALS = [
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "xi",
    "xii",
    "xiii",
    "xiv",
    "xv",
    "xvi",
]


@main.command()
@click.option("--as-list/--no-as-list", default=False)
def dump_all_algs(as_list: bool):
    paths = get_paths()
    results = get_results(paths)
    all_ops = {}
    for r in results:
        for alg, op in zip(
            itertools.chain.from_iterable(r["winning_algorithms"]),
            itertools.chain.from_iterable(r["matching_operations"]),
        ):
            if alg not in all_ops:
                all_ops[alg] = set([op])
            else:
                all_ops[alg].add(op)

    sorted_keys = sorted(all_ops.keys())

    if as_list:
        print(sorted_keys)
    else:
        for i, alg in enumerate(sorted(all_ops.keys())):
            print(f"{ROMAN_NUMERALS[i]} & {alg} & {len(all_ops[alg])} \\\\")


# taken from result of dump_all_algs
SORTED_ALGS = [
    # direct loop
    "Volta fused conv/ReLU",
    "grouped naive kernel",
    # GEMM
    "Ampere implicit gemm",
    # "Ampere fprop implicit gemm", # commented out because not present in table
    "Kepler implicit sgemm",
    # "Kepler precomputed sgemm",
    "explicit sgemm",
    "implicit sgemm",
    "precomputed sgemm",
    # winograd
    "Ampere Winograd",
    "Maxwell Winograd",
    "Maxwell Winograd nonfused",
    "Turing Winograd nonfused",
    "Volta compiled winograd",
    "Volta nonfused Winograd",
    # FFT
    "FFT gemm",
]


@main.command()
def deterministic_float():
    paths = [
        "results/winning_algorithm/local/server-GTX_1650-cifar10_medium_instrumental-cifar10_medium_instrumental-i0.json",
        "results/winning_algorithm/local/consumer-RTX_2070-cifar10_medium_instrumental-cifar10_medium_instrumental-i0.json",
    ]
    results = get_results(paths)

    algorithms = []
    for r in results:
        winning_algorithms = r["winning_algorithms"]
        unique_winnings = []
        for run in winning_algorithms:
            if run not in unique_winnings:
                unique_winnings.append(run)
        algorithms.append(unique_winnings)
    # HACK for now
    algorithms[0] = algorithms[0][0]
    algorithms[1] = algorithms[1][1]

    dump_gpu_tikz(algorithms)

    # just for the text
    predictions = [
        [
            metadata.convert_json_to_np(p["output"])
            for p in r["all_predictions"][0]
            if "conv" in p["layer_name"]
        ]
        for r in results
    ]
    assert all(p1.shape == p2.shape for p1, p2 in zip(*predictions))
    assert len(predictions) == 2
    diverging = [np.any(p1 != p2) for p1, p2 in zip(*predictions)]


@main.command()
def indeterministic_float():
    path = "results/winning_algorithm/gcloud/nvidia_tesla_p100-Tesla_P100-PCIE-16GB-cifar10_medium_instrumental-i0.json"
    result = get_results([path])[0]
    unique_algorithms = []
    for algorithm in result["winning_algorithms"]:
        if algorithm not in unique_algorithms:
            unique_algorithms.append(algorithm)

    dump_gpu_tikz(unique_algorithms)

    # just for the text
    predictions = [
        [
            metadata.convert_json_to_np(p["output"])
            for p in ps
            if "conv" in p["layer_name"]
        ]
        for ps in result["all_predictions"]
    ]

    for i, ps in enumerate(zip(*predictions)):
        uniques = []
        eqcs = [compare.get_equivalence_class(uniques, p) for p in ps]
        print(f"% {i}: {max(eqcs)+1} eqcs")


@main.command()
def compare_all_gpus():
    paths = [p for p in get_paths() if "medium_instrumental-i0" in p]
    results = get_results(paths)

    convolution_indices = [
        i for i, name in enumerate(results[0]["layer_names"]) if "conv" in name
    ]

    for result in results:
        result["winning_algorithm"] = result["winning_algorithms"][0]
        result["intermediate_results"] = [
            metadata.convert_json_to_np(l["output"])
            for l in result["all_predictions"][0]
            if "conv" in l["layer_name"]
        ]

    def get_eqcs(predictions):
        uniques = []
        return [compare.get_equivalence_class(uniques, p) for p in predictions]

    num_eqcs = [
        max(get_eqcs(column))
        for column in zip(*[r["intermediate_results"] for r in results])
    ]
    y_values = {alg: i for i, alg in enumerate(SORTED_ALGS)}

    rows = []
    for r in results:
        rows += [
            {
                "layer_idx": i,
                "alg": y_values[alg],
                "device_name": metadata.get_clean_device_name(r),
            }
            for i, alg in enumerate(r["winning_algorithm"])
        ]
    df = pd.DataFrame(rows)
    print(num_eqcs)
    sns.relplot(
        data=df, x="layer_idx", y="alg", hue="device_name", kind="line", marker="."
    )
    plt.yticks(ticks=range(len(SORTED_ALGS)), labels=SORTED_ALGS)
    plt.show()


def dump_gpu_tikz(algorithms):
    alg_coordinates = {
        op: len(SORTED_ALGS) - i - 0.7 for i, op in enumerate(SORTED_ALGS)
    }
    for i, y in enumerate(alg_coordinates.values()):
        if i % 2:
            print(
                f"\\fill[black!20] (-3.2,{y}-.5) rectangle ({len(algorithms[0])}, {y}+.5);"
            )
        print(
            f"\\draw (-0.3,{y}) node[left,align=center,minimum width=1cm] {{ {ROMAN_NUMERALS[i]} }};"
        )
    print(
        f"\\draw (-0.3,{max(alg_coordinates.values())}+1.5) node[left,align=center] (ylab) {{ Conv.\\\\ alg. }};"
    )
    print(
        f"\\draw[thick] (-3.2,{max(alg_coordinates.values())}+0.42) -- (ylab.east |- 0,{max(alg_coordinates.values())}+0.42) -- node[above] {{Algorithm choice per layer}} (20,{max(alg_coordinates.values())}+0.42);"
    )
    # print(f"\\draw[thick] (ylab.south east) -- (ylab.south west);")

    for i, algs in enumerate(algorithms):
        print(f"\\begin{{scope}}[style{i}]")
        print(f"\\draw")
        for j, alg in enumerate(algs):
            print(
                f"{'--' if j > 0 else ''} ({j}, {alg_coordinates[alg]}) node[datapoint] {{}}"
            )
        print(";")
        print(f"\\end{{scope}}")

    # print(f"\\draw[->,thick] (-.2,-0.5) -- (-.2,{len(SORTED_ALGS)-0.5});")
    print(
        f"\\draw[->,thick] (-.4,-.2) -- ++({len(algorithms[0])}+.2, 0) node[midway,below] {{ \\scriptsize Convolution layers }};"
    )
    # print(f"\\draw[thick] (-.4,-.2) -- ++(0,16);")
    print(
        "\n".join(
            [
                f"\\draw[thin] ({i},-.2) -- ++(0,-.1);"
                for i, _ in enumerate(algorithms[0])
            ]
        )
    )


def assert_global_alg_determinism(results: List[Dict]) -> None:
    rows = []
    for result in results:
        model_type = result["model_type"]
        sample_index = result["sample_index"]
        device_name = metadata.get_clean_device_name(result)

        for run_algs, run_predictions in zip(
            result["winning_algorithms"], result["all_predictions"]
        ):
            alg_string = "-".join(run_algs)
            final_prediction = run_predictions[-1]["output"]["bytes"]
            rows.append(
                {
                    "model_type": model_type,
                    "sample_index": sample_index,
                    "device_name": device_name,
                    "alg_string": alg_string,
                    "prediction": final_prediction,
                }
            )

    df = pd.DataFrame(rows)
    num_eqcs = df.groupby(
        ["model_type", "sample_index", "device_name", "alg_string"]
    ).nunique()
    assert not any(num_eqcs.prediction > 1)


def get_algs_per_card(results: List[Dict]) -> Dict[str, Set]:
    card_algorithms = {}
    for result in results:
        winning_algorithms = result["winning_algorithms"]
        device = metadata.get_clean_device_name(result)
        flat_algs = set(flatten(winning_algorithms))
        try:
            current_set = card_algorithms[device]
        except KeyError:
            current_set = set()
        card_algorithms[device] = current_set | flat_algs

    return card_algorithms


def sanity_check(result: Dict) -> None:
    all_winning_algorithms = result["winning_algorithms"]
    all_predictions = result["all_predictions"]

    unique_predictions = {}
    for run_algs, run_predictions in zip(all_winning_algorithms, all_predictions):
        final_prediction = run_predictions[-1]
        alg_string = "-".join(run_algs)

        try:
            assert final_prediction == unique_predictions[alg_string]
        except KeyError:
            unique_predictions[alg_string] = final_prediction


def generate_divergence_traces(result: Dict):
    all_winning_algorithms = result["winning_algorithms"]
    all_predictions = result["all_predictions"]

    runs = [
        build_run(algs, predictions)
        for algs, predictions in zip(all_winning_algorithms, all_predictions)
    ]

    divergence_traces = build_divergence_traces(runs)
    target_name, _ = splitext(basename(result["path"]))
    dot_string = generate_trace_dot(divergence_traces)

    dot_directory = join(RESULT_DIR, "dots")
    os.makedirs(dot_directory, exist_ok=True)
    with open(join(dot_directory, f"{target_name}.dot"), "w") as f:
        f.write(dot_string)


def generate_trace_dot(divergence_traces):
    REASON_MAPPING = {
        DivergenceReason.INPUT: "diff. Input",
        DivergenceReason.ALGORITHM: "diff. Algorithm",
        DivergenceReason.UNKNOWN: "UNKNOWN",
    }

    def generate_name(step_idx, eqc):
        return f"node_{step_idx}_{eqc}"

    combined_traces = combine_divergence_traces(divergence_traces)
    ret = "digraph graph_name {\n"

    ret += "//nodes\n"
    # print all equivalence classes as nodes
    for i, step in enumerate(combined_traces):
        for eqc in sorted(step.keys()):
            name = generate_name(i, eqc)
            algs = step[eqc]["algs"]
            label = f"EQC: {eqc}\\n" + "\\n".join(algs)
            ret += f'{name} [label="{label}"];\n'

    ret += "\n"

    edges = {}
    for i, column in enumerate(zip(*divergence_traces)):
        if i == 0:
            continue

        for j, data_point in enumerate(column):
            current_eqc, _, reason = data_point
            old_eqc = divergence_traces[j][i - 1][0]

            vertices = (generate_name(i - 1, old_eqc), generate_name(i, current_eqc))
            try:
                edges[vertices].add(reason)
            except KeyError:
                edges[vertices] = set([reason])

    ret += "//edges\n"
    for (source, sink), reasons in edges.items():
        reasons = filter(lambda r: r is not None, reasons)
        label = "\n".join([REASON_MAPPING[reason] for reason in reasons])
        ret += f'{source} -> {sink} [label="{label}"];\n'
    ret += "\n"
    ret += "}"

    return ret


def build_run(winning_algorithms: List, predictions: List) -> List:
    # fold inputs into convolution layers
    for before, current in zip(itertools.chain([{}], predictions), predictions):
        if "conv" not in current["layer_name"].lower():
            continue

        conv_input = before["output"]
        current["input"] = conv_input

    conv_predictions = [
        prediction
        for prediction in predictions
        if "conv" in prediction["layer_name"].lower()
    ]
    assert len(conv_predictions) == len(winning_algorithms)  # both are of same length

    for prediction, algorithm in zip(conv_predictions, winning_algorithms):
        prediction["algorithm"] = algorithm

    return conv_predictions


def build_divergence_traces(runs: List[List]) -> List:
    equivalence_sets = [set(range(len(runs)))]

    def find_containing_eqc_set(idx: int) -> Set:
        """find the eqc_set containing run index idx"""
        for eqc_set in equivalence_sets:
            if idx in eqc_set:
                return eqc_set

        raise ValueError("Did not find containing eqc set")

    divergence_traces = []
    for _ in runs:
        divergence_traces.append([])

    for conv_idx, data_points in enumerate(zip(*runs)):
        # iterate over convolutions (layers/columns)

        still_fitting = []
        need_moving = []
        for run_idx, data_point in enumerate(data_points):
            # iterate over data_points (runs/rows)

            containing_set = find_containing_eqc_set(run_idx)

            # check if our prediction is the same as ALL others in the same set
            def check_divergence(data_point, eqc_set) -> Union[DivergenceReason, None]:
                for other_idx in eqc_set:
                    other_data_point = data_points[other_idx]
                    if data_point["input"] != other_data_point["input"]:
                        return DivergenceReason.INPUT
                    if data_point["output"] != other_data_point["output"]:
                        if data_point["algorithm"] != other_data_point["algorithm"]:
                            return DivergenceReason.ALGORITHM
                        else:
                            return DivergenceReason.UNKNOWN

                return None  # we haven't diverged

            # if we haven't diverged, append to the respective trace
            divergence_reason = check_divergence(data_point, containing_set)
            if divergence_reason is None:
                still_fitting.append(run_idx)
            else:
                containing_set.remove(run_idx)
                need_moving.append((run_idx, divergence_reason))

        assert len(need_moving) + len(still_fitting) == len(
            runs
        )  # ensure all values are accounted for

        # move everything that needs to be moved
        for run_idx, reason in need_moving:
            data_point = data_points[run_idx]
            # try to find a new home
            divergence_reasons = [
                check_divergence(data_point, eqc_set) for eqc_set in equivalence_sets
            ]
            try:
                target_set_idx = divergence_reasons.index(None)
                equivalence_sets[target_set_idx].add(run_idx)
            except ValueError:
                # make a new home
                equivalence_sets.append(set([run_idx]))
                target_set_idx = len(equivalence_sets) - 1
            divergence_traces[run_idx].append(
                (target_set_idx, data_points[run_idx]["algorithm"], reason)
            )

        # add traces for everything that can stay
        for run_idx in still_fitting:
            set_idx = equivalence_sets.index(find_containing_eqc_set(run_idx))
            divergence_traces[run_idx].append(
                (set_idx, data_points[run_idx]["algorithm"], None)
            )

    return divergence_traces


def combine_divergence_traces(divergence_traces):
    ret = []
    for column in zip(*divergence_traces):
        combined = {}
        for eqc, alg, reason in column:
            try:
                combined[eqc]["algs"].add(alg)
                combined[eqc]["reasons"].add(reason)
            except KeyError:
                combined[eqc] = {"algs": set([alg]), "reasons": set([reason])}

        ret.append(combined)

    return ret


def get_paths(glob_exp=""):
    glob_term = join(RESULT_DIR, "*", f"*{glob_exp}*.json")
    paths = glob(glob_term, recursive=True)
    paths = [path for path in paths if "instrumental" in path]
    return paths


def load_file(path: str):
    unpack_keys = []
    with open(path, "r") as f:
        result = json.load(f)
        for key in unpack_keys:
            result[key] = metadata.convert_json_to_np(result[key])

        return result


def analyze_result(result: Dict) -> Dict:
    runs = result["convolution_operations"]

    all_algorithms = []
    all_operations = []
    for run in runs:
        winning_algorithms = []
        matching_operations = []

        for conv in run:
            calls = conv["function_calls"]

            try:
                winning_algorithm, operation = metadata.get_winning_algorithm(
                    calls, True
                )
            except AssertionError as e:
                previous_message = str(e)
                raise AssertionError(
                    f"{previous_message}\nDevice is {result['device']['physical_description']['name']}"
                )

            winning_algorithms.append(winning_algorithm)
            matching_operations.append(operation)

        all_algorithms.append(winning_algorithms)
        all_operations.append(matching_operations)

    ret = result.copy()
    ret["winning_algorithms"] = all_algorithms
    ret["matching_operations"] = all_operations
    return ret


if __name__ == "__main__":
    main()
    sys.exit(0)
