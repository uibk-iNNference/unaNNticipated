from glob import glob
from typing import Dict
import json
import eval_definitions as defs
from posixpath import join, basename, splitext
from innfrastructure import compare
from functools import reduce
import itertools
import numpy as np
from joblib import Parallel, delayed, parallel_backend

RESULT_DIR = join("results", "equivalence_classes")


def main():
    paths = get_paths()
    results = [load_result(path) for path in paths]
    results = [
        r
        for r in results
        if r["device"]["device_type"].lower() == "cpu"
        and r["model_type"] == "cifar10_medium"
        and r["sample_index"] == 0
    ]

    preprocess(results)

    # print_flag_sets(results)

    # reduce_relevant_flags(results) # do manually, takes forever this way

    confusion_matrix = get_confusion_matrix(results)
    print_confusion_matrix(confusion_matrix)


def get_paths():
    glob_term = join(RESULT_DIR, "*", "cpu", "*.json")
    return glob(glob_term)


def load_result(path: str) -> Dict:
    with open(path, "r") as input_file:
        result = json.load(input_file)

        flags = frozenset(result["cpu_info"]["flags"])
        result["flags"] = flags

        filename = splitext(basename(path))[0]
        sample_index_text = filename.split("-")[-1]
        sample_index = int(sample_index_text[1:])
        result["sample_index"] = sample_index

        return result


def preprocess(results):
    unique_predictions = []
    unique_cpu_stats = []
    for result in results:
        append_eqcs(unique_predictions, unique_cpu_stats, result)


def append_eqcs(unique_predictions, unique_cpu_stats, result):
    eqc = compare.get_equivalence_class(unique_predictions, result["prediction"])

    cpu_info = result["cpu_info"]
    relevant_flags = set(cpu_info["flags"]) & defs.RELEVANT_FLAGS
    cpu_stats = (
        frozenset(relevant_flags),
        cpu_info["count"],
    )
    cpu_eqc = compare.get_equivalence_class(unique_cpu_stats, cpu_stats)

    result["equivalence_class"] = eqc
    result["cpu_stats_class"] = cpu_eqc


def print_flag_sets(results):
    flags_per_eqc = build_flags_per_eqc(results)

    irrelevant_flags = get_irrelevant_flags(flags_per_eqc)
    print("Flags not relevant for eqcs (both present and not present in same eqc):")
    print(", ".join(sorted(irrelevant_flags)))
    print()

    full_sets = [
        reduce(lambda x, y: x | y, flag_sets, set())
        for flag_sets in flags_per_eqc.values()
    ]
    full_diff_set = set()
    for set_a, set_b in itertools.combinations(full_sets, 2):
        current_diff_set = set_a ^ set_b
        full_diff_set = full_diff_set | current_diff_set
    print("Union over all pairwise diff sets between eqcs:")
    print("Union(a ^ b for a,b, in combinations(flags_per_eqc))")
    print(", ".join(sorted(full_diff_set)))
    print()

    print("Full diff set without irrelevant_flags:")
    print(", ".join(sorted(full_diff_set - irrelevant_flags)))
    print()


def build_flags_per_eqc(results):
    flags_per_eqc = {}
    for result in results:
        eqc = result["equivalence_class"]
        try:
            flags_per_eqc[eqc].append(result["flags"])
        except KeyError:
            flags_per_eqc[eqc] = [result["flags"]]

    return flags_per_eqc


def get_irrelevant_flags(flags_per_eqc):
    irrelevant_flags = set()
    for eqc, flag_sets in flags_per_eqc.items():
        current_flags = set()

        for flags in flag_sets[1:]:
            diff_set = flags ^ flag_sets[0]
            current_flags = current_flags | diff_set

        if len(current_flags) > 0:
            irrelevant_flags = irrelevant_flags | current_flags
            # print(f"Irrelevant flags for eqc {eqc}:")
            # print(", ".join([flag for flag in sorted(current_flags)]))

    return irrelevant_flags


def reduce_relevant_flags(results):
    full_filter_set = defs.RELEVANT_FLAGS

    identifiers = [
        (
            r["equivalence_class"],
            (r["cpu_info"]["count"], frozenset(r["cpu_info"]["flags"])),
        )
        for r in results
    ]

    for filter_set_size in range(len(full_filter_set)):
        print(f"Filter set size: {filter_set_size}")
        with parallel_backend("loky"):
            results = Parallel()(
                delayed(check_collision_freeness)(filter_flags, identifiers)
                for filter_flags in itertools.combinations(
                    full_filter_set, filter_set_size
                )
            )
        if any(map(lambda r: r[0], results)):
            print(f"Found collision free filter set of size {filter_set_size}")
            for collision_free, filter_set in results:
                if collision_free:
                    print(filter_set)


def check_collision_freeness(filter_flags, identifiers):
    filter_set = set(filter_flags)
    unique_cpu_stats = []
    cpu_stat_eqc_mappings = {}
    collision = False

    for eqc, (core_count, flags) in identifiers:
        current_cpu_stats = (core_count, flags & filter_set)
        cpu_stat_class = compare.get_equivalence_class(
            unique_cpu_stats, current_cpu_stats
        )

        try:
            assert eqc == cpu_stat_eqc_mappings[cpu_stat_class]
        except KeyError:
            cpu_stat_eqc_mappings[cpu_stat_class] = eqc
        except AssertionError:
            collision = True
            return False, None

    if not collision:
        return True, filter_set


def get_confusion_matrix(results):
    unique_eqcs = []

    results = [
        r.copy()
        for r in results
        if r["model_type"] == "cifar10_medium" and r["sample_index"] == 0
    ]
    preprocess(results)

    eqcs = [result["equivalence_class"] for result in results]
    max_eqc = max(eqcs) + 1
    max_cpu_eqc = max([result["cpu_stats_class"] for result in results]) + 1
    confusion_matrix = np.zeros((max_cpu_eqc, max_eqc), dtype=bool)

    eqc_mapping = get_sorted_eqc_mapping(results)
    for result in results:
        eqc = result["equivalence_class"]
        mapped_eqc = eqc_mapping[eqc]
        result["equivalence_class"] = mapped_eqc

    for result in results:
        cpu_eqc = result["cpu_stats_class"]
        eqc = result["equivalence_class"]
        confusion_matrix[cpu_eqc, eqc] = True

    assert all(confusion_matrix.sum(axis=1) == 1)
    for eqc in eqcs:
        matching_results = [r for r in results if r["equivalence_class"] == eqc]
        flag_sets = set(
            [
                frozenset(r["cpu_info"]["flags"]) & defs.RELEVANT_FLAGS
                for r in matching_results
            ]
        )

        diff_union = reduce(
            set.union,
            map(lambda a, b: a ^ b, itertools.combinations(flag_sets, 2)),
            set(),
        )
        if len(diff_union) > 0:
            print(f"EQC: {eqc}")
            print(diff_union)
    return confusion_matrix


def print_confusion_matrix(confusion_matrix):
    confusion_matrix = confusion_matrix.astype(int)
    for row in confusion_matrix:
        row_string = "  ".join(map(lambda v: str(v), row))
        row_string = row_string.replace("0", "_")
        print(row_string)


def get_sorted_eqc_mapping(results):
    eqcs = set([r["equivalence_class"] for r in results])
    uniques = [True] * len(eqcs)

    for eqc in eqcs:
        result_subset = [r for r in results if r["equivalence_class"] == eqc]
        for r in result_subset[1:]:
            if any(
                map(
                    lambda r: r["cpu_stats_class"] != result_subset[0],
                    result_subset[1:],
                )
            ):
                uniques[eqc] = False

    sorted_eqc_indices = np.argsort(uniques)
    eqc_mapping = {
        eqc: target_index
        for eqc, target_index in zip(sorted_eqc_indices, range(len(sorted_eqc_indices)))
    }
    return eqc_mapping


def print_common_flags():
    paths = glob('results/equivalence_classes/*/cpu/*json', recursive=True)
    results = [json.load(open(path, 'r')) for path in paths]
    flag_sets = [set(r['cpu_info']['flags']) for r in results]
    print(reduce(lambda a,b: a&b, flag_sets))


if __name__ == "__main__":
    # main()
    print_common_flags()
