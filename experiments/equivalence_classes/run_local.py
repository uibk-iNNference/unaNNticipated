from common import gpu_experiment, cpu_experiment
from joblib import parallel_backend, Parallel, delayed
from innfrastructure import remote
from posixpath import join
import os
import sys

# CPU_HOSTS = ["consumer", "rechenknecht", "server"]
CPU_HOSTS = ["rechenknecht"]
GPU_HOSTS = ["consumer"]
RESULT_DIR = join("results", "equivalence_classes", "local")


def run_cpu():
    result_dir = join(RESULT_DIR, "cpu")
    os.makedirs(result_dir, exist_ok=True)

    params = []
    for host in CPU_HOSTS:
        remote.prepare_server(host)
        hostname = host
        # if host == "rechenknecht":
        #     hostname += "_6gb"

        params.append((host, hostname, result_dir))

    with parallel_backend("loky"):
        Parallel()(delayed(cpu_experiment)(*param) for param in params)


def run_gpu():
    result_dir = join(RESULT_DIR, "gpu")
    os.makedirs(result_dir, exist_ok=True)

    for host in GPU_HOSTS:
        remote.prepare_server(host)

    with parallel_backend("loky"):
        Parallel()(
            delayed(gpu_experiment)(host, host, result_dir) for host in GPU_HOSTS
        )


if __name__ == "__main__":
    if "--gpu" in sys.argv or "-g" in sys.argv:
        run_gpu()

    if "--cpu" in sys.argv or "-c" in sys.argv:
        run_cpu()
