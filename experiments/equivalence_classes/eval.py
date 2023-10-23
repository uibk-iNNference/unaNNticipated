import copy

import click
import numpy as np
from glob import glob
import json
from innfrastructure import metadata, compare
from posixpath import join, basename
import pandas as pd
import sys
from operator import itemgetter
from functools import reduce
import eval_definitions as defs
import sanity_checks
from typing import List

RESULT_DIR = join("results", "equivalence_classes")


def get_memory_size(path, result):
    try:
        if "gpu" in path:
            num_bytes = int(result["device"]["memory_limit"])
        else:
            num_bytes = int(result["memory"]["total"])
    except KeyError:
        return "ERROR"

    num_gib = num_bytes / 1024 / 1024 / 1024
    return f"{num_gib:.1f}"


def get_device_manufacturer(device_name: str):
    if "amd" in device_name.lower():
        return "AMD"
    if "intel" in device_name.lower():
        return "Intel"
    return "NVidia"


def shorten_device_name(device_name: str):
    device_name = device_name.replace("AMD EPYC", "").replace("Intel", "")
    if "Xeon" in device_name:
        parts = device_name.split()
        if len(parts) > 1:
            # it's a Xeon processor with number designation, cut off Xeon
            device_name = " ".join([p for p in parts[1:] if not p.startswith("v")])

    prefix = "Tesla "
    if device_name.startswith(prefix):
        device_name = device_name[len(prefix):]
    return device_name


def get_microarchitecture(
    manufacturer: str, device_name: str, cpu_family: int, cpu_model: int
):
    if manufacturer.lower() == "nvidia":
        return defs.NVIDIA_MICROARCHITECTURES[device_name]

    if manufacturer.lower() == "intel":
        return defs.INTEL_MICROARCHITECTURES[cpu_model]

    if manufacturer.lower() == "amd":
        return defs.AMD_MICROARCHITECTURES[(cpu_family, cpu_model)]

    raise ValueError(f"Unknown manufacturer {manufacturer}")


def load_result(path: str):
    with open(path, "r") as f:
        result_dict = json.load(f)

    result_dict["prediction"] = metadata.convert_json_to_np(result_dict["prediction"])

    sample_index_part = path.split("-")[-1].split(".")[0]
    sample_index = int(sample_index_part[1:])
    result_dict["sample_index"] = sample_index

    if "gcloud" in path:
        cloud = "gcloud"
    elif "aws" in path:
        cloud = "aws"
    elif "local" in path:
        cloud = "local"
    result_dict["cloud"] = cloud

    instance_type = basename(path).split("-")[0]
    result_dict["instance_type"] = instance_type

    result_dict["path"] = path

    result_dict["device_type"] = "gpu" if "gpu" in path else "cpu"

    device_name = metadata.get_clean_device_name(result_dict).replace("_", " ")
    result_dict["device_name"] = shorten_device_name(device_name)

    manufacturer = get_device_manufacturer(device_name)
    result_dict["manufacturer"] = manufacturer

    cpu_family = int(result_dict["cpu_info"]["family"])
    cpu_model = int(result_dict["cpu_info"]["model"])
    microarchitecture = get_microarchitecture(
        manufacturer, device_name, cpu_family, cpu_model
    )
    result_dict["microarchitecture"] = microarchitecture

    if "coffee" in microarchitecture.lower():
        cpu_model = 105  # HACK to get coffee lake before ice lake

    try:
        stepping = result_dict["cpu_info"]["stepping"]
    except KeyError:
        stepping = -1
    result_dict["cpu_stepping"] = stepping

    result_dict["model_sort_index"] = defs.MODEL_TYPE_SORT_INDEX[
        result_dict["model_type"]
    ]

    # device specific stuff
    if manufacturer.lower() == "nvidia":
        device_sort_index = defs.NVIDIA_SORTING[microarchitecture]
        core_count = "NA"
    else:
        device_sort_index = cpu_model
        core_count = result_dict["cpu_info"]["count"]

    result_dict["device_sort_index"] = device_sort_index
    result_dict["core_count"] = core_count

    result_dict["memory"] = get_memory_size(path, result_dict)

    # check if count is sufficient
    result_dict["deterministic"] = True
    try:
        count = result_dict["count"]
        all_predictions = metadata.convert_json_to_np(result_dict["all_predictions"])
        result_dict["deterministic"] = count == all_predictions.shape[0]

        if count < 0.6 * all_predictions.shape[0]:
            ratio = count / all_predictions.shape[0]
            _, counts = np.unique(all_predictions, return_counts=True, axis=0)
            counts = sorted(counts, reverse=True)
            print(
                f"WARNING: most common prediction only happened in {ratio * 100}% of cases for path {path}\ncounts: {counts}",
                file=sys.stderr,
            )

    except KeyError:
        pass

    return result_dict


def get_pivot_table():
    paths = [
        p
        for p in glob(join(RESULT_DIR, "**", "*.json"), recursive=True)
        if "cifar10_large" not in p
    ]
    results = [load_result(path) for path in paths]
    results = sorted(
        results,
        key=itemgetter(
            "model_sort_index",
            "sample_index",
            "manufacturer",
            "core_count",
            "device_sort_index",
            "device_name",
        ),
    )

    unique_predictions = []
    unique_flag_sets = []
    unique_cpu_stats = []
    rows = [
        extract_df_row(unique_predictions, unique_flag_sets, unique_cpu_stats, result)
        for result in results
    ]
    df = pd.DataFrame(rows)
    sanity_checks.assert_all(df)

    pivot = df.pivot(
        index=[
            "instance_type",
            "manufacturer",
            "microarchitecture",
            "core_count",
            "memory_size",
            "device_name",
            "cloud",
            # "cpu_family",
            "device_sort_index",
            "flag_set_class",
            "cpu_stats",
        ],
        columns=["model_type", "sample_index"],
        values=["equivalence_class", "deterministic"],
    )

    pivot = pivot.sort_values(["manufacturer", "core_count", "device_sort_index"])
    # check pivot table for missing data
    missing = pivot.isna().any(axis=1)
    assert not any(missing)

    return pivot


def extract_df_row(
    unique_predictions: List, unique_flag_sets: List, unique_cpu_stats: List, result
):
    eqc = compare.get_equivalence_class(unique_predictions, result["prediction"])

    core_count = result["cpu_info"]["count"]
    flag_set = (frozenset(set(result["cpu_info"]["flags"]) & defs.RELEVANT_FLAGS),)
    flag_set_class = compare.get_equivalence_class(unique_flag_sets, flag_set)

    cpu_stats = metadata.extract_cpu_stats(result)
    cpu_stat_class = compare.get_equivalence_class(unique_cpu_stats, cpu_stats)

    return {
        "model_type": result["model_type"],
        "model_sort_index": defs.MODEL_TYPE_SORT_INDEX[result["model_type"]],
        "device_sort_index": result["device_sort_index"],
        "sample_index": result["sample_index"],
        "equivalence_class": eqc,
        "flag_set_class": flag_set_class,
        "cpu_stats": cpu_stats,
        "cpu_stat_class": cpu_stat_class,
        "hostname": result["hostname"],
        "device_name": result["device_name"],
        "device_type": result["device_type"],
        "cloud": result["cloud"],
        "manufacturer": result["manufacturer"],
        "microarchitecture": result["microarchitecture"],
        "cpu_family": f'0x{result["cpu_info"]["family"]:x}',
        "cpu_model": f'{result["cpu_info"]["model"]:03}',
        "instance_type": result["instance_type"],
        "path": result["path"],
        "cpu_stepping": result["cpu_stepping"],
        "memory_size": result["memory"],
        "core_count": result["core_count"],
        "deterministic": result["deterministic"],
    }


def get_rounded_count(cores: int) -> int:
    borderline_counts = [1, 2, 4, 8, 9]
    rounded_count = None
    for borderline_count in borderline_counts:
        if cores >= borderline_count:
            rounded_count = borderline_count
    assert rounded_count is not None

    return rounded_count


@click.group()
def main():
    pass


@main.command()
def html():
    pivot = get_pivot_table()
    # pivot = pivot.sort_values(("equivalence_class", "cifar10_medium", 0))
    print(pivot.to_html())

@main.command()
def cumulative():
    pivot = get_pivot_table()
    pivot = pivot.sort_values(["manufacturer", "microarchitecture", "core_count", "device_name"])
    pivot = pivot.sort_values(("equivalence_class", "cifar10_medium", 1))

    dict_values = {}
    for model in ['cifar10_small', 'cifar10_medium', 'deep_weeds']:
        eqcs = pivot['equivalence_class'][model][1].values
        seen_eqcs = set()
        cumulative_count = []
        for eqc in eqcs:
            seen_eqcs.add(eqc)
            cumulative_count.append(len(seen_eqcs))

        dict_values[model] = cumulative_count

    dict_values['index'] = range(1,len(cumulative_count)+1)
    df = pd.DataFrame(dict_values)
    print(df.to_csv())


CLOUD_NAMES = {"gcloud": "GCP", "aws": "AWS", "local": "local"}

COLUMN_NAMES = [
    "vendor",
    "microarchitecture",
    "cores",
    "memory",
    "device_name",
    "cloud",
    "cpu_class",
    "instance_type",
    "small_0",
    "small_1",
    "small_6",
    "medium_0",
    "medium_1",
    "medium_6",
    "large_0",
    "large_1",
    "large_6",
]


@main.command()
@click.option("--long-version/--no-long-version", type=bool)
def latex(long_version: bool):
    pivot = get_pivot_table()
    # pivot = pivot.sort_values(("equivalence_class", "cifar10_medium", 0), kind='stable')
    table = build_latex_table(pivot)
    clean_table(table, long_version)
    table = post_process_table(table)
    dump_table(table, long_version)


EQC_START = 8


def build_latex_table(pivot):
    table = []
    for row_index in pivot.index:
        (
            instance_type,
            manufacturer,
            microarchitecture,
            core_count,
            memory_size,
            device_name,
            cloud,
            device_sort_index,
            flag_set_class,
            cpu_stat_class,
        ) = row_index

        table_row = {
            "vendor": manufacturer,
            "cores": core_count,
            "microarchitecture": microarchitecture,
            "device_name": device_name,
            "memory": memory_size,
            "cloud": cloud,
            "cpu_class": cpu_stat_class,
            "instance_type": instance_type,
        }

        sub_df = pivot.loc[row_index]
        for i, sub_index in enumerate(sub_df.index):
            if "deterministic" in sub_index:
                break  # HACK: this is dependent on the order of values in the pivot() call

            eqc = sub_df[sub_index]
            deterministic = sub_df[("deterministic",) + sub_index[1:]]

            col_name = COLUMN_NAMES[i + EQC_START]
            table_row[col_name] = {"eqc": eqc, "deterministic": deterministic}

        table.append(table_row)
    return table


def clean_table(table, long_version):
    max_values = []
    initial_values = []
    for v in list(table[0].values())[EQC_START:]:
        max_values.append(-1)
        initial_values.append(v["eqc"])

    for i, row in enumerate(table):
        row["cloud"] = CLOUD_NAMES[row["cloud"]]

        device_name = row["device_name"]
        words = device_name.split(" ")
        new_words = []
        for word in words:
            if word.lower() == "platinum":
                continue

            if "GB" in word:  # NVidia enterprise cards
                word = word.split("-")[0]

            new_words.append(word)

        row["device_name"] = " ".join(new_words)

        for j, name in enumerate(COLUMN_NAMES[EQC_START:]):
            current_dict = row[name]
            current_dict["eqc"] -= initial_values[j]
            current_dict["bold"] = False
            current_dict["gray"] = False

            if current_dict["eqc"] > max_values[j]:
                max_values[j] = current_dict["eqc"]
                current_dict["bold"] = True

            if long_version and j > 0:
                current_dict["gray"] = (
                    row[COLUMN_NAMES[j - 1 + EQC_START]]["eqc"] == current_dict["eqc"]
                )

def dump_table(table, long_version):
    row_strings = []
    for i, row in enumerate(table):
        row_copy = clean_row(row, long_version)
        row_string = " & ".join(map(str, row_copy.values()))
        if i < len(table) - 1:
            row_string += "\\\\"  # last line requires a break outside the \input
        row_string += f"% {row['instance_type']}"
        row_strings.append(row_string)
        print(row_string)

def clean_row(row, long_version):
    row_copy = copy.deepcopy(row)
    del row_copy["instance_type"]

    # shorten for double column version
    if not long_version:
        del row_copy['small_0']
        del row_copy['small_1']
        del row_copy['small_6']
        del row_copy['large_1']
        del row_copy['large_6']
        del row_copy['medium_0']
        del row_copy['medium_6']

    return row_copy

def post_process_table(table):
    seen_content = []
    unique_cpu_class = []

    output_rows = []
    row_index = 0
    for row in table:
        new_row = copy.deepcopy(row)

        new_row["cpu_class"] = compare.get_equivalence_class(
            unique_cpu_class, new_row["cpu_class"]
        )

        # filter duplicates, workaround with all the post processing
        output_content = []
        for key, value in new_row.items():
            if key == "instance_type":
                continue
            if isinstance(value, dict):
                value = value["eqc"]
            output_content.append(value)
        if output_content in seen_content:
            continue
        seen_content.append(output_content)

        for name in COLUMN_NAMES[EQC_START:]:
            current_dict = row[name]
            value = current_dict["eqc"]

            value = (
                "\\hphantom{*}"
                + str(value)
                + ("\\hphantom{*}" if current_dict["deterministic"] else "*")
            )
            if current_dict["bold"]:
                value = f"\\textbf{{{value}}}"
            if current_dict["gray"]:
                value = f"\\textcolor{{uibkgray!80}}{{{value}}}"

            new_row[name] = value

        if row_index % 5 == 0:
            new_row[COLUMN_NAMES[0]] = (
                "\\tikz[overlay]{\\draw (-.4,0.5ex) node {\\tiny(%d)};}"
                % (row_index + 1)
                + list(row.values())[0]
            )

        output_rows.append(new_row)
        row_index += 1

    return output_rows


if __name__ == "__main__":
    main()
