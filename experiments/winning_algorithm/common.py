import gzip
import itertools
import json
import os
import sys
import tempfile
from datetime import datetime
from glob import glob
from io import BytesIO
from posixpath import join
from typing import Dict, List

import invoke
import numpy as np
from fabric import Connection
from innfrastructure import metadata, models, remote
from innfrastructure.gcloud import GcloudConnection
from joblib import Parallel, delayed, parallel_backend
from tensorflow import keras
from dataclasses import dataclass


@dataclass
class RunConfiguration:
    model_name: str
    sample_path: str
    sample_indices: List[int]
    result_dir: str


@dataclass
class HostConfiguration:
    address: str
    hostname: str
    user: str = None
    connection_options: Dict = None


@dataclass
class MeasurementInfo:
    layer_names: List[str]
    convolution_layer_indices: List[int]
    num_convolutions: int
    model_name: str
    model_storage_path: str


def download_logs(host: str, logdir: str) -> str:
    """Download traces from remote host, and return the path to the gzipped json file on local

    Args:
        host (str): The remote hostname
        logdir (str): The log directory on the remote (as passed to innfcli)

    Returns:
        str: the path to the file on the local machine
    """
    basedir = os.path.dirname(logdir)
    invoke.run(f"rm -rf {logdir}")
    print(f"Downloading {logdir}")
    invoke.run(f"rsync -avz {host}:'{logdir}' '{basedir}/'")
    glob_term = join(logdir, "**", "*trace.json.gz")
    traces = glob(glob_term, recursive=True)

    assert len(traces) == 1
    return traces[0]


def get_measurement_info(model: keras.Model, instrumental_model_dir: str):
    layer_names = [layer.name for layer in model.layers]
    convolution_layer_indices = [
        i for i, n in enumerate(layer_names) if "conv" in n.lower()
    ]
    num_convolutions = len(convolution_layer_indices)
    model_name = model.name
    model_storage_path = join(instrumental_model_dir, f"{model_name}.h5")

    return MeasurementInfo(
        layer_names,
        convolution_layer_indices,
        num_convolutions,
        model_name,
        model_storage_path,
    )


def build_instrumental_model(target_model: str, target_dir: str) -> str:
    print(f"Building instrumental for {target_model}")

    # get model and layer
    model = models.get_model(target_model)
    instrumental_model = keras.models.Model(
        model.inputs[0],
        [layer.output for layer in model.layers],
        name=model.name + "_instrumental",
    )
    target_path = join(target_dir, f"{instrumental_model.name}.h5")
    instrumental_model.save(target_path)

    return instrumental_model


def parse_operations(path):
    with gzip.open(path, "r") as unzipped:
        trace = json.load(unzipped)
    events = filter(
        lambda e: "ts" in e.keys()
        and not (e["name"].startswith("Memory") or e["name"].startswith("$")),
        trace["traceEvents"],
    )
    events = sorted(events, key=lambda e: e["ts"])
    events = [e["name"] for e in events]

    operations = []
    current_op = "setup"
    current_functions = []
    for event in events:
        if event.lower().startswith("eagerexecute"):
            operations.append({"name": current_op, "function_calls": current_functions})
            current_op = event.split(" ")[1]
            current_functions = []

        # if event.lower().startswith("void"):
        current_functions.append(event)

    return operations


def run_prediction(
    connection, host, sample_index, all_convolutions, all_predictions, measurement_info
):
    with tempfile.TemporaryDirectory(
        prefix=f"{host.hostname}-{sample_index}-"
    ) as logdir:
        out_path = join(logdir, "result.json")
        cmd = f"innfcli predict {measurement_info.model_name} /tmp/sample.npy -d /device:GPU:0 -b -p {logdir} --output_path {out_path}"
        connection.docker_run(cmd, hide=True)
        connection.get(out_path, out_path)
        with open(out_path, "r") as result_file:
            result = json.load(result_file)
        connection.docker_run(f"rm {out_path}")  # delete before we copy the entire directory

        # filter for convolutions
        trace_path = download_logs(host.address, logdir)
        operations = parse_operations(trace_path)
        convolutions = [o for o in operations if o["name"].lower() == "conv2d"]
        convolutions = convolutions[-measurement_info.num_convolutions :]
        all_convolutions.append(convolutions)

        # add prediction
        current_predictions = result["prediction"]
        filtered_predictions = [
            {"index": i, "layer_name": measurement_info.layer_names[i], "output": r}
            for i, r in enumerate(current_predictions)
            if i in measurement_info.convolution_layer_indices
            or i + 1
            in measurement_info.convolution_layer_indices  # to check precondition of convolutions
            or i == len(measurement_info.layer_names) - 1  # for final output
        ]

        all_predictions.append(filtered_predictions)

        # add sample_index
        result["sample_index"] = sample_index

    return result


def experiment_per_sample(
    connection,
    samples,
    sample_index,
    num_datapoints,
    host,
    measurement_info,
    run_config,
):
    current_sample = samples[sample_index : sample_index + 1]
    current_sample = np.vstack((current_sample, current_sample))
    f = BytesIO()
    np.save(f, current_sample)
    print(f"Sending sample number {sample_index} to host {host.hostname}...")
    connection.put(f, "/tmp/sample.npy")
    experiment_time_stamp = datetime.now()

    predictions = []
    all_convolutions = []

    for i in range(num_datapoints):
        print(f"Prediction {i} on {host.hostname}")
        result = run_prediction(
            connection,
            host,
            sample_index,
            all_convolutions,
            predictions,
            measurement_info,
        )

    del result["prediction"]  # remove to save space
    result["all_predictions"] = predictions
    result["convolution_operations"] = all_convolutions
    result["layer_names"] = measurement_info.layer_names

    cleaned_host_name = host.hostname.replace("-", "_")
    device_name = metadata.get_clean_device_name(result)
    target_path = join(
        run_config.result_dir,
        f"{cleaned_host_name}-{device_name}-{measurement_info.model_name}-i{sample_index}.json",
    )
    with open(target_path, "w") as target_file:
        print(f"Saving result in {target_path}...")
        json.dump(result, target_file)


def run_per_host(
    host: HostConfiguration,
    num_datapoints: int,
    measurement_info: MeasurementInfo,
    run_config: RunConfiguration,
):
    with GcloudConnection(
        host.address, host.user, connect_kwargs=host.connection_options
    ) as connection:
        print(
            f"Performing GPU experiment for {measurement_info.model_name} on {host.hostname}..."
        )

        # transfer in two stages to get the model into the container
        transfer_result = connection.put(
            measurement_info.model_storage_path,
            "/tmp/",
        )
        connection.docker_run(f"mv {transfer_result.remote} /forennsic/data/models")

        samples = np.load(run_config.sample_path)
        for sample_index in run_config.sample_indices:
            experiment_per_sample(
                connection,
                samples,
                sample_index,
                num_datapoints,
                host,
                measurement_info,
                run_config,
            )


def run_on_all_hosts(
    run_configuration: RunConfiguration, hosts: List[HostConfiguration]
):
    os.makedirs(run_configuration.result_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="winning_alg_model_dir") as model_dir:
        print(run_configuration.model_name)
        instrumental_model = build_instrumental_model(
            run_configuration.model_name, model_dir
        )
        measurement_info = get_measurement_info(instrumental_model, model_dir)

        with parallel_backend("loky"):
            Parallel()(
                delayed(run_per_host)(host, 33, measurement_info, run_configuration)
                for host in hosts
            )
        # [run_per_host(host, 33, measurement_info, run_configuration) for host in hosts]
