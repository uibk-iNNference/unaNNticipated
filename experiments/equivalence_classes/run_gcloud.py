import sys

from innfrastructure import gcloud
from common import cpu_experiment, gpu_experiment
from joblib import parallel_backend, Parallel, delayed
from posixpath import join
import os
import itertools

from innfrastructure.gcloud import GcloudConnection

RESULT_DIR = join("results", "equivalence_classes", "gcloud")
GPU_RESULT_DIR = join(RESULT_DIR, "gpu")
CPU_RESULT_DIR = join(RESULT_DIR, "cpu")

CPU_CONFIGS = [
    "intel-sandy-bridge",
    "intel-ivy-bridge",
    "intel-haswell",
    "intel-broadwell",
    "intel-skylake",
    "intel-ice-lake",
    "intel-cascade-lake",
    "amd-rome",
    "amd-milan",
]

GPU_CONFIGS = [
    "nvidia-t4",
    "nvidia-k80",
    "nvidia-v100",
    "nvidia-p100",
    "nvidia-a100",
]


def experiment_per_host(instance):
    with GcloudConnection(instance.ip) as conn:
        if "intel" in instance.name or "amd" in instance.name:
            cpu_experiment(conn, instance.name, CPU_RESULT_DIR)

        if "nvidia" in instance.name:
            gpu_experiment(conn, instance.name, 20, GPU_RESULT_DIR)


def run(configs):
    gcloud.ensure_configs_running(configs)
    instances = [instance for instance in gcloud.get_instances() if instance.name in configs]
    assert len(instances) == len(configs)
    with parallel_backend("loky", n_jobs=8):
        Parallel()(delayed(experiment_per_host)(instance) for instance in instances)

    # gcloud.cleanup_configs(configs)


def main():
    os.makedirs(GPU_RESULT_DIR, exist_ok=True)
    os.makedirs(CPU_RESULT_DIR, exist_ok=True)

    if "-c" in sys.argv or "--cpu" in sys.argv:
        substrings = ['', '-4','-8','-16','-32']
        configs = list(map(lambda a: f"{a[0]}{a[1]}", itertools.product(CPU_CONFIGS, substrings)))
        run(configs)

    if "-g" in sys.argv or "--gpu" in sys.argv:
        run(GPU_CONFIGS)


if __name__ == "__main__":
    main()
