import gzip
import json
from glob import glob
from os.path import basename, join, splitext
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow
import seaborn as sns
import streamlit as st

BASE_DIR = join("results", "secondary", "profiler")


def get_paths():
    return glob(join(BASE_DIR, "**", "*trace.json*"), recursive=True)


def filter_trace_name(path):
    with gzip.open(path, "r") as f:
        traces = json.load(f)

    return [event["name"] for event in traces["traceEvents"] if "name" in event.keys()]


def print_trace_lengths(paths):
    trace_names = []
    for e, path in enumerate(paths):
        trace_names.append(filter_trace_name(path))

        dir = Path(path).parts
        description = dir[4]

        print(f"Number of traces for {description}: {len(trace_names[e])}")


def load_similarities():
    paths = glob(join(BASE_DIR, "similarities", "*.feather"))

    dfs = [pd.read_feather(path) for path in paths]

    data = {}
    for path, df in zip(paths, dfs):
        model_type = splitext(basename(path))[0]
        header = model_type.replace("_", " ").title()

        df = df.set_index("name")
        df = df.reindex(columns=sorted(df.columns))

        data[header] = df

    return data


def plot_similarities():
    st.markdown(
        """
    ## Similarity between traces

    Below is series of similarity measures between traces.
    The traces are taken using the TensorFlow profiler, and can also be viewed in TensorBoard using the Trace viewer.

    Before analysis we cleaned the traces by filtering out all trace events that do not have a timestamp.

    A similarity of 1 indicates identical traces (after cleanup).
    If all our CPU traces have a similarity of 1, then the execution is indiscernable as far as TensorFlow is concerned.
    """
    )
    data = load_similarities()
    headers = ["Minimal"]  # TODO: add other model_types

    for header in headers:
        df = data[header]
        st.markdown(f"### {header}")
        st.dataframe(df)


def load_diffs():
    paths = glob(join(BASE_DIR, "similarities", "*.feather"))
    data = {}
    for path in paths:
        try:
            data[path] = pd.read_feather(path)
        except pyarrow.lib.ArrowInvalid:
            # empty diff
            continue


def main():
    # print_trace_lengths(get_paths())
    plot_similarities()


if __name__ == "__main__":
    main()
