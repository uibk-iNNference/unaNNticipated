# NH: Eine konkrete Frage zu prepare_eval: warum hasch du die Trace Names in ints umgewandelt?

from difflib import get_close_matches
import glob
from os.path import join
import os


def compare_traces():
    """
    Get symmetric differences of traces from two devices.
    (Traces that are in device a but not in device b, and vice versa.)
    """
    paths = get_paths()
    for path_a in paths:
        traces_a = load_traces(path_a)
        for path_b in paths:
            traces_b = load_traces(path_b)

            # differences = traces_a["traces"].symmetric_difference(traces_b["traces"])
            traces_a["diff_ab"] = traces_a["traces"].difference(traces_b["traces"])
            traces_b["diff_ba"] = traces_b["traces"].difference(traces_a["traces"])

            # map_traces(traces_a, traces_b)
            save_comparison(traces_a, traces_b)


def get_paths():
    return glob.glob("results/secondary/profiler/traces/*.txt")


def map_traces(traces_a, traces_b):
    if traces_a["diff_ab"]:
        filtered_numbered_traces_a = set()
        for trace in traces_a["diff_ab"]:
            filtered_numbered_traces_a.add(
                get_respective_trace(trace, traces_a["traces_numbered"])[0]
            )

        traces_a["diff_ab"] = filtered_numbered_traces_a

    if traces_b["diff_ba"]:
        filtered_numbered_traces_b = set()
        for trace in traces_b["diff_ba"]:
            filtered_numbered_traces_b.add(
                get_respective_trace(trace, traces_b["traces_numbered"])[0]
            )

        traces_b["diff_ba"] = filtered_numbered_traces_b


# map the traces from ["traces_numbered"] with the once from ["diff_ab"] and ["diff_ba"]


def load_traces(path):
    with open(path) as f:
        traces = f.readlines()
        # adding numbered traces, in order to map them later.
        traces_numbered = []
        for i, trace in enumerate(traces):
            traces_numbered.append(join(str(i), trace))

    filename = path.split("/")[4]

    traces = {
        "host": filename[0 : filename.find("-")],
        "device": path.split("-")[1],
        "traces": clean_traces(traces),
        "traces_numbered": clean_traces(traces_numbered),
    }

    return traces


def clean_traces(traces):
    """
    Often times the traces differ, but only in the parameter "addr:" or "correlation_id".
    I assume, those parameters are only metainformation and not relevant for the analysis. This should be checked though.
    A cleaned trace has no parameter "addr:" or "correlation_id".
    Parameters where differences appear: "occupancy_min_grid_size", "num_bytes", "theoretical_occupancy_pct", "occupancy_suggested_block_size", "is_eager"

    """
    substrings = ["addr:", "correlation_id"]
    cleaned_traces = []
    for trace in traces:
        for substring in substrings:
            if substring in trace:
                start_rmv = trace.find(substring)
                end_rmv = trace.find(",", start_rmv)
                trace = join(trace[0:start_rmv:] + trace[end_rmv + 1 : :])

        cleaned_traces.append(trace)
    return set(cleaned_traces)


def save_comparison(traces_a, traces_b):
    """
    Layout:

    a: Host-GPU
    b: Host-GPU

    a: trace_a
    b: respective trace_b
    """

    result_dir = join("results", "secondary", "profiler", "differences")
    os.makedirs(result_dir, exist_ok=True)

    filename = f'{traces_a["host"]}-{traces_a["device"]}-{traces_b["host"]}-{traces_b["device"]}.txt'

    with open(join(result_dir, filename), "w") as filehandle:
        filehandle.write(f'a: {traces_a["host"]}-{traces_a["device"]}\n')
        filehandle.write(f'b: {traces_b["host"]}-{traces_b["device"]}\n')

        diffs_b = traces_b["diff_ba"]
        for diff in traces_a["diff_ab"]:
            diff_a = diff

            try:
                diff_b = get_respective_trace(diff_a, traces_b["diff_ba"])[0]
                diffs_b.remove(diff_b)
            except IndexError:
                diff_b = "\n"

            filehandle.write("\n a: %s" % diff_a)
            filehandle.write(" b: %s" % diff_b)

        if diffs_b:
            filehandle.write("\n Traces in b only:\n")
            for diff in diffs_b:
                filehandle.write(" b: %s" % diff)

    print(
        f'{traces_a["host"]}-{traces_a["device"]}:{len(traces_a["diff_ab"])} when comparing with {traces_b["host"]}-{traces_b["device"]}:{len(traces_b["diff_ba"])}'
    )


def get_respective_trace(trace_a, traces_b):
    """comparing two traces, only returns the closest match. Cutoff is default 0.6"""
    return get_close_matches(trace_a, traces_b, 1)


def main():
    compare_traces()


if __name__ == "__main__":
    main()
