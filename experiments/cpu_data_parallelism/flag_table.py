import functools
import itertools
import json
import math
from glob import glob
from os.path import basename
from typing import List

import click
import pandas as pd

from innfrastructure import compare

MACHINE_NAMES = [
    "intel_sandy_bridge",
    "intel_ivy_bridge",
    "intel_haswell",
    "intel_broadwell",
    "intel_skylake",
    "intel_ice_lake",
    "intel_cascade_lake",
    "amd_rome",
    "amd_milan",
]
SIMD_FLAGS = ["avx2", "avx512f", "avx512vbmi", "sse4a"]

# functools.reduce(lambda a,b: a+b, [flag_cluster_contents[i] for i in variate_within])
IRRELEVANT_FLAGS = [
    "avx512vnni",
    "avx512_vnni",
    "ibrs_enhanced",
    "3dnowprefetch",
    "adx",
    "rdseed",
    "smap",
    "erms",
    "pcid",
    "f16c",
    "fsgsbase",
    "rdrand",
    "rdrnd",
    "smep",
    "hle",
    "rtm",
    "invpcid",
    "invpcid_single",
    "mpx",
    "pti",
]


def get_sort_index(target: str, candidates: List[str]) -> int:
    for i, filename in enumerate(candidates):
        if basename(target).startswith(filename):
            return i
    return i + 1


# if __name__ == "__main__":
def generate_flag_clusters():
    paths = sorted(
        [
            p
            for p in glob("results/equivalence_classes/gcloud/cpu/*json")
            if any(
                [basename(p).startswith(candidate + "-") for candidate in MACHINE_NAMES]
            )
            and "cifar10_medium-i0" in p
        ],
        key=lambda p: get_sort_index(p, MACHINE_NAMES),
    )
    results = [json.load(open(p, "r")) for p in paths]
    cpu_flag_configurations = [frozenset(r["cpu_info"]["flags"]) for r in results]

    constant_flags = functools.reduce(frozenset.intersection, cpu_flag_configurations)
    relevant_flags = [f - constant_flags for f in cpu_flag_configurations]

    # combine flags that always occur together
    all_flags = sorted(
        sorted(
            functools.reduce(frozenset.union, relevant_flags)  # sort alphabetically
        ),
        key=lambda f: get_sort_index(f, SIMD_FLAGS)
        if f not in IRRELEVANT_FLAGS
        else 1000,
    )
    flag_clusters = {}
    for flag in all_flags:
        occurences = tuple(flag in s for s in cpu_flag_configurations)
        if occurences in flag_clusters:
            flag_clusters[occurences].append(flag)
        else:
            flag_clusters[occurences] = [flag]
    return (
        flag_clusters,
        constant_flags,
        cpu_flag_configurations,
        all_flags,
        paths,
        results,
    )


@click.group()
def main():
    pass


@main.command()
def df():
    (
        flag_clusters,
        constant_flags,
        cpu_flag_configurations,
        all_flags,
        paths,
        results,
    ) = generate_flag_clusters()

    flag_cluster_contents = list(flag_clusters.values())

    # assert set correctness
    # if any flag of a set is contained in a set
    # all should be contained
    for flag_cluster in cpu_flag_configurations:
        for flag_cluster in flag_clusters.values():
            contained = [f in flag_cluster for f in flag_cluster]
            assert any(contained) == all(contained), flag_cluster
    # no flags unaccounted for
    unaccounted_flags = [
        f for f in all_flags if not any([f in c for c in flag_clusters.values()])
    ]
    assert len(unaccounted_flags) == 0

    # for each result check which sets are present
    rows = {}
    for i, (path, result) in enumerate(zip(paths, results)):
        instance_name = basename(path).split("-")[0]
        row = {j: c[i] for j, c in enumerate(flag_clusters.keys())}
        rows[instance_name] = row

    df = pd.DataFrame(rows)
    visual_df = pd.DataFrame([df[column].map({True: "X ", False: ""}) for column in df])

    # relate these flag set changes to EQCs
    # generate eqcs
    unique_predictions = []
    eqcs = {
        str(name): compare.get_equivalence_class(
            unique_predictions, r["prediction"]["bytes"]
        )
        for name, r in zip(df.columns, results)
    }
    df_eqcs = [eqcs[k] for k in visual_df.index]
    visual_df["eqc"] = df_eqcs
    visual_df = visual_df.sort_values("eqc")

    # find relevant sets
    def get_sub_df(eqc):
        names = [name for name, value in eqcs.items() if value == eqc]
        return df[names]

    # remove all sets that are both present and not present WITHIN an EQC
    variate_within = set()
    for eqc in set(eqcs.values()):
        sub_df = get_sub_df(eqc)
        irrelevant = sub_df[
            sub_df.apply(
                lambda x: any(x) != all(x), axis=1
            )  # present in one, missing in other
        ].index
        variate_within |= set(irrelevant)

    visual_df = visual_df[["eqc"] + list(range(len(flag_cluster_contents)))]
    print(visual_df)


@main.command()
def full_table():
    (
        flag_clusters,
        constant_flags,
        cpu_flag_configurations,
        all_flags,
        paths,
        results,
    ) = generate_flag_clusters()

    # print flags and their containing cluster
    print()
    really_all_flags = sorted(
        list(functools.reduce(frozenset.union, cpu_flag_configurations))
    )

    flag_cluster_contents = list(flag_clusters.values())

    def get_flag_cluster_index(flag, flag_cluster_contents):
        if flag is None:
            return ""
        for i, cluster in enumerate(flag_cluster_contents):
            if flag in cluster:
                return i
        return "\\textit{C}"

    def clean_flag_name(flag_name: str) -> str:
        return flag_name.replace("_", "\\_") if flag_name else ""
    
    num_columns = 3
    column_height = math.ceil(len(really_all_flags) / num_columns)
    columns = itertools.zip_longest(*[iter(really_all_flags)] * column_height) # the splat operator calls next() sequentially, stepping each subsequent copy of the iterator forward as well
    for row in zip(*columns):
        value_tuples = [(clean_flag_name(flag), get_flag_cluster_index(flag, flag_cluster_contents)) for flag in row]
        print(" & ".join([f"\\texttt{{{flag}}} & {cluster_index}" for flag, cluster_index in value_tuples]) + "\\\\")

    print("\\bottomrule") # latex doesn't like it when you put a bottomrule after an input, so we do it here

if __name__ == "__main__":
    main()
