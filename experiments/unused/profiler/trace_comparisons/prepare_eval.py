import gzip
import json
from glob import glob
import os
from os.path import join, dirname
from typing import List
from joblib import parallel_backend, Parallel, delayed

import pandas as pd
from Levenshtein.StringMatcher import StringMatcher

BASE_DIR = join(
    "results",
    "secondary",
    "profiler",
)

MODEL_TYPES = [
    "minimal",
]
# for size in ["small", "medium", "large"]:
#     MODEL_TYPES.append(f"cifar10_{size}")


def get_paths():
    return glob(join(BASE_DIR, "**", "*trace.json*"), recursive=True)


def load_trace(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "r") as f:
            return json.load(f)

    print(f"WARNING: skipping already extracted trace at {path}")


def clean_trace(trace):
    # filter out events that don't have a timestamp
    # (they mess up analysis, and are probably not relevant to execution)
    events = [event for event in trace["traceEvents"] if "ts" in event.keys()]

    # sort by start time
    sorted_events = sorted(events, key=lambda e: e["ts"])

    # we only care about names for now
    cleaned_events = []
    for event in sorted_events:
        cleaned_event = event["name"]

        if "args" in event.keys():
            cleaned_event += str(event["args"])

        cleaned_events.append(cleaned_event)

    return cleaned_events


def build_encoding(traces: List[List[str]]):
    encoding = {}
    decoding = {}
    current_val = 0
    encoded_traces = []

    for trace in traces:
        encoded_trace = []

        for name in trace:

            try:
                value = encoding[name]
            except KeyError:
                value = current_val
                encoding[name] = value
                decoding[chr(value)] = name
                current_val += 1

            encoded_value = chr(value)
            encoded_trace.append(encoded_value)

        encoded_traces.append("".join(encoded_trace))

    return encoded_traces, decoding


def main():
    with parallel_backend("loky"):
        Parallel()(delayed(prepare_model)(model_type) for model_type in MODEL_TYPES)


def prepare_model(model_type):
    filtered_paths = [path for path in get_paths() if model_type in path]
    traces = [load_trace(path) for path in filtered_paths]

    cleaned_traces = [clean_trace(trace) for trace in traces]
    store_traces(filtered_paths, cleaned_traces)

    encoded_traces, decoding = build_encoding(cleaned_traces)

    # prepare paths for naming
    trace_names = [build_trace_name(path) for path in filtered_paths]

    trace_dict = {name: trace for name, trace in zip(trace_names, encoded_traces)}

    # build_similarities(model_type, trace_dict)
    # build_diffs(model_type, decoding, trace_dict)


def store_traces(filtered_paths, cleaned_traces):
    for path, trace in zip(filtered_paths, cleaned_traces):
        path_parts = path.split("/")
        identifier = path_parts[4]

        dirname = join(BASE_DIR, "traces")
        os.makedirs(dirname, exist_ok=True)

        target_name = join(dirname, f"{identifier}.txt")
        with open(target_name, "w") as target_file:
            target_file.writelines("\n".join(trace))


def build_trace_name(path):
    parts = path.split("/")
    identifier = parts[-6]
    hostname, device, _ = identifier.split("-")
    trace_name = "-".join([hostname, device])
    return trace_name


def build_similarities(model_type, trace_dict):
    target_path = get_similarity_path(model_type)

    df = calculate_similarities(trace_dict)
    os.makedirs(dirname(target_path), exist_ok=True)
    df.reset_index(inplace=True)
    df.to_feather(target_path)

    return df


def get_similarity_path(model_type):
    return join(BASE_DIR, "similarities", f"{model_type}.feather")


def calculate_similarities(trace_dict):
    rows = []
    for name, trace in trace_dict.items():
        current_row = {"name": name}
        for other_name, other_trace in trace_dict.items():
            if name == other_name:
                continue

            matcher = StringMatcher(seq1=trace, seq2=other_trace)
            current_row[other_name] = matcher.ratio()
        rows.append(current_row)

    df = pd.DataFrame(rows)
    return df


def build_diffs(model_type, decoding, trace_dict):
    for name, trace in trace_dict.items():
        for other_name, other_trace in trace_dict.items():
            if name == other_name:
                continue

            build_diff(model_type, name, trace, other_name, other_trace, decoding)


def build_diff(model_type, name, trace, other_name, other_trace, decoding):
    target_path = get_diff_path(model_type, name, other_name)
    df = calculate_diff(trace, other_trace, decoding)
    os.makedirs(dirname(target_path), exist_ok=True)

    if len(df) == 0:
        with open(target_path, "w") as f:
            f.write("# traces were identical\n")
        return None

    df.reset_index(inplace=True)
    df.to_feather(target_path)

    return df


def get_diff_path(model_type, name, other_name):
    return join(BASE_DIR, "diffs", f"{model_type}-{name}-{other_name}.feather")


def calculate_diff(trace, other_trace, decoding):
    # simplify analysis by reporting traces of unequal length now
    # if len(trace) != len(other_trace):
    #     print(f"Traces {name} and {other_name} are not of equal length")

    i = 0
    diffs = []
    while True:
        try:
            event = trace[i]
        except IndexError:
            # print(f"Trace {name} ended")
            break

        try:
            other_event = other_trace[i]
        except IndexError:
            # print(f"Trace {other_name} ended")
            break

        if event != other_event:
            diffs.append(
                {
                    "location": i,
                    "event": decoding[event],
                    "other_event": decoding[other_event],
                }
            )

        i += 1

    df = pd.DataFrame(diffs)
    return df


if __name__ == "__main__":
    main()
