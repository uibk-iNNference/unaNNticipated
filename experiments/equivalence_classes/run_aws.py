import sys
import time

from innfrastructure import aws
from common import cpu_experiment, gpu_experiment
from posixpath import join
from joblib import parallel_backend, Parallel, delayed
import os

from innfrastructure.aws import AwsConnection

RESULT_DIR = join("results", "equivalence_classes", "aws")
GPU_RESULT_DIR = join(RESULT_DIR, "gpu")
CPU_RESULT_DIR = join(RESULT_DIR, "cpu")

CPU_CONFIGS = [[
    # "t3.large",
    # "t2.medium",
    # "t2.large",
    # "t2.xlarge",
    "m5.large",
    # "r5.large",
    # "m4.large",
    # "c4.xlarge",
    # "c5.xlarge",
    # "c5n.xlarge",
    # "x1e.xlarge",
    # "z1d.large",
    # "i3en.large",
],
    [
        "c5.12xlarge",
    ]]

GPU_CONFIGS = [[
    # "p2.xlarge",
    "g4dn.xlarge",
],
    [
        "g3s.xlarge",
        # "p3.2xlarge",
        # "p3dn.24xlarge",
        # "p4d.24xlarge",
    ]]


def experiment_per_host(instance):
    instance_type = instance.name[0]
    with AwsConnection(instance.ip) as connection:
        if instance_type in ["p", "g"]:
            gpu_experiment(
                connection,
                instance.name,
                10,
                GPU_RESULT_DIR,
            )
        else:
            cpu_experiment(
                connection,
                instance.name,
                CPU_RESULT_DIR,
            )


def run(configs):
    for shard in configs:
        aws.ensure_configs_running(shard)
        time.sleep(10)  # AWS starts instances async, so we need to wait for them to exist
        instances = [instance for instance in aws.get_running_instances() if instance.name in shard]
        assert len(instances) == len(shard)

        with parallel_backend("loky"):
            Parallel()(delayed(experiment_per_host)(instance) for instance in instances)

        aws.cleanup_configs(shard)


def main():
    os.makedirs(CPU_RESULT_DIR, exist_ok=True)
    os.makedirs(GPU_RESULT_DIR, exist_ok=True)

    if "-c" in sys.argv or "--cpu" in sys.argv:
        run(CPU_CONFIGS)
    if "-g" in sys.argv or "--gpu" in sys.argv:
        run(GPU_CONFIGS)


if __name__ == "__main__":
    main()
