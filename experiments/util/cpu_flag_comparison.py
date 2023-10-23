import glob
import json
import click
from invoke import Context
from invoke.exceptions import UnexpectedExit
from innfrastructure import metadata
from typing import List

RELEVANT_FLAGS = [
    "ADX",
    "AES",
    "AVX2",
    "AVX",
    "AVX512F",
    "AVX512CD",
    "AVX512PF",
    "AVX512ER",
    "AVX512VL",
    "AVX512BW",
    "AVX512DQ",
    "AVX512VBMI",
    "AVX512IFMA",
    "AVX512_4VNNIW",
    "AVX512_4FMAPS",
    "BMI1",
    "BMI2",
    "CMOV",
    "CMPXCHG16B",
    "CMPXCHG8B",
    "F16C",
    "FMA",
    "MMX",
    "PCLMULQDQ",
    "POPCNT",
    "PREFETCHW",
    "PREFETCHWT1",
    "RDRAND",
    "RDSEED",
    "SMAP",
    "SSE2",
    "SSE3",
    "SSE4_1",
    "SSE4_2",
    "SSE",
    "SSSE3",
    "HYPERVISOR",
]  # taken from tensorflow/tensorflow/core/platform/cpu_info.cc


def get_flags(paths: List[str]):
    flags_dict = {}
    for path in paths:
        with open(path) as f:
            result_dict = json.load(f)
            flags = result_dict["cpu_info"]["flags"]
            host = result_dict["hostname"]
            prediction = result_dict["prediction"]["bytes"]
            flags_dict[host] = {"prediction": prediction, "flags": flags}

    return flags_dict


@click.group()
def main():
    pass


@main.command("filter")
@click.argument("paths", nargs=-1)
@click.option("--search-command", default="rg -i -c")
@click.option("--tensorflow-dir", default="~/Projects/tensorflow/")
def filter_flags_in_diff(paths: List[str], search_command: str, tensorflow_dir: str):
    flag_diffs = compare_flags(paths)
    for flag_diff in flag_diffs:
        for name, diffs in flag_diff["differences"].items():
            matching_terms = [term for term in diffs if term.upper() in RELEVANT_FLAGS]
            print(f"{name}: {matching_terms}")


@main.command("print")
@click.argument("paths", nargs=-1)
def print_flag_diffs(paths: List[str]):
    flag_diffs = compare_flags(paths)
    differences = [host["differences"] for host in flag_diffs]
    print(json.dumps(differences, indent=2))


def compare_flags(paths: List[str]):
    flags_dict = get_flags(paths)
    flags_diffs = []

    for host in flags_dict:
        comparison_dict = {}

        for next_host in flags_dict:
            if host == next_host:
                continue

            flags_host = set(flags_dict[host]["flags"])
            flags_next_host = set(flags_dict[next_host]["flags"])
            host_diff_next_host = list(flags_host.difference(flags_next_host))

            compared_hosts = f"{host}/{next_host}"
            comparison_dict[compared_hosts] = host_diff_next_host

            flags_diffs.append(
                {
                    "hostname": host,
                    "flags": list(flags_host),
                    "prediction": flags_dict[host]["prediction"],
                    "differences": comparison_dict,
                }
            )
    return flags_diffs


if __name__ == "__main__":
    main()
