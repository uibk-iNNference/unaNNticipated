import sys
import json
from typing import Set
import eval_definitions as defs


def assert_all(df):
    assert_device_eqc_uniqueness(df)
    assert_no_meaningful_diffs(df)
    #  assert_eqc_cpu_stat_uniqueness(df)
    assert_cpu_stat_eqc_uniqueness(df)
    assert_flags_eqc_non_uniqueness(df)


def assert_device_eqc_uniqueness(df):
    # ensure that all results with the same device have the same equivalence_class
    unique = df.groupby(["model_type", "sample_index", "device_name", "instance_type"])
    assert all(unique.equivalence_class.max() == unique.equivalence_class.min())


def assert_no_meaningful_diffs(df):
    # get devices which result in multiple equivalence_classes
    by_device = df.groupby(["model_type", "sample_index", "device_name"])
    differing_eqcs = (
        by_device.equivalence_class.max() != by_device.equivalence_class.min()
    )
    target_rows = differing_eqcs[differing_eqcs].reset_index()

    meaningful_diff = False
    for row in target_rows.iloc:
        # get values for each index column in current row
        key = tuple([row[k] for k in row.index.values[:-1]])
        original_indices = by_device.groups[key]
        datapoints = [df.iloc[i] for i in original_indices]

        # some exclusion criteria
        cpu_models = [dp.cpu_model for dp in datapoints]
        if any([m != cpu_models[0] for m in cpu_models]):
            continue  # brand_raw is the same for multiple generations

        memory_sizes = [dp.memory_size for dp in datapoints]
        if any([mem != memory_sizes[0] for mem in memory_sizes]) or any(
            [mem == "ERROR" for mem in memory_sizes]
        ):
            continue  # different memory sizes

        sorted_instance_types = list(sorted([dp.instance_type for dp in datapoints]))
        if sorted_instance_types == ["c5_12xlarge", "c5_xlarge"]:
            # New C5 and C5d 12xlarge, 24xlarge, and metal instance sizes feature custom 2nd generation Intel Xeon Scalable Processors
            # [source: https://aws.amazon.com/ec2/instance-types/]
            # these custom CPUs seem to use the same family and model ID as the others, but have more extensions (AVX512vnni)
            continue

        if (
            "t2_medium" in sorted_instance_types
            and "t2_xlarge" in sorted_instance_types
        ):
            # something to do with memory size, I'm working on it
            continue

        diff_paths = ""
        for datapoint in datapoints:
            print(
                f"CPU family: {datapoint.cpu_family}, CPU model: {datapoint.cpu_model}"  # remember stepping is a thing
            )
            diff_paths += f"<(jq '.' '{datapoint.path}') "
        print(f"Diff command:\nvimdiff {diff_paths}")
        print("\n")

        meaningful_diff = True

    if meaningful_diff:
        raise AssertionError(
            "There were differing EQCs not accounted for or previously known"
        )


def assert_eqc_cpu_stat_uniqueness(df):
    subset = df.query("device_type == 'cpu' and model_type == 'cifar10_medium'")
    grouped = subset.groupby(["model_type", "sample_index", "equivalence_class"])
    non_uniques = grouped["cpu_stat_class"].nunique() > 1
    # get non unique indices
    indices = list(non_uniques[non_uniques].dropna().index)

    if len(indices) == 0:
        return

    for index in indices:
        print(
            f"Have non_unique cpu stats in equivalence_class {index}",
            file=sys.stderr,
        )

    raise AssertionError


def assert_cpu_stat_eqc_uniqueness(df):
    subset = df.query("device_type == 'cpu' and model_type == 'cifar10_medium'")
    grouped = subset.groupby(["model_type", "sample_index", "cpu_stat_class"])
    non_uniques = grouped["equivalence_class"].nunique() > 1
    # get non unique indices
    indices = list(non_uniques[non_uniques].dropna().index)

    if len(indices) == 0:
        return

    for index in indices:
        print(
            f"Have non-unique equivalence class for cpu stats {index}",
            file=sys.stderr,
        )
        print(file=sys.stderr)

    raise AssertionError


def assert_flags_eqc_non_uniqueness(df):
    subset = df.query("device_type == 'cpu'")
    grouped = subset.groupby(["model_type", "sample_index", "flag_set_class"])
    non_uniques = grouped["equivalence_class"].nunique() > 1
    indices = list(non_uniques[non_uniques].dropna().index)

    assert len(indices) > 0

def get_flags(path: str) -> Set:
    with open(path, "r") as input_file:
        result = json.load(input_file)
        flags = set(result["cpu_info"]["flags"])
        return flags - defs.RELEVANT_FLAGS
