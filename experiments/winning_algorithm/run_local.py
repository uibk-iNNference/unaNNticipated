import os
import sys
import tempfile
from posixpath import join

from common import run_on_all_hosts, RunConfiguration, HostConfiguration
from fabric import Connection
from innfrastructure import models
from joblib import Parallel, delayed, parallel_backend
from dataclasses import dataclass
from typing import List

RESULT_DIR = join("results", "winning_algorithm", "local")


def main():
    run_configurations = []
    for size in ["small", "medium"]:
        run_configurations.append(
            RunConfiguration(
                f"cifar10_{size}",
                join("data", "datasets", "cifar10", "samples.npy"),
                [0, 1, 2],
                RESULT_DIR,
            )
        )

    host_configurations = []
    # for hostname in ["consumer", "rechenknecht", "server"]:
    for hostname in ["server"]:
        host_configurations.append(HostConfiguration(hostname, hostname))

    for run_config in run_configurations:
        run_on_all_hosts(run_config, host_configurations)


if __name__ == "__main__":
    main()
