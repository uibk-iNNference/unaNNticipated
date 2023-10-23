from invoke import run
from joblib.parallel import delayed
from main import run as main
from secondary.deterministic_cuda import run as deterministic_cuda
from secondary.placement import run as placement
from secondary.profiler import run as profiler

from joblib import Parallel, parallel_backend, delayed
from innfrastructure import remote

hosts = ["rechenknecht", "ennclave-consumer", "ennclave-server"]


def experiment_per_host(host, run_cpu, run_gpu):
    print(f"\n\nRunning main on {host}\n\n")
    main.run(host, run_cpu, run_gpu)
    print(f"\n\nRunning deterministic_cuda on {host}\n\n")
    deterministic_cuda.run(host)
    print(f"\n\nRunning placement on {host}\n\n")
    placement.run(host)
    print(f"\n\nRunning profiler on {host}\n\n")
    profiler.run(host, run_cpu, run_gpu)


if __name__ == "__main__":
    import sys

    run_cpu = "--run_cpu" in sys.argv
    run_gpu = "--run_gpu" in sys.argv

    if not run_cpu and not run_gpu:
        print("ERROR: you didn't specify any devices to run on")
        sys.exit(1)

    with parallel_backend("loky", n_jobs=len(hosts)):
        Parallel()(delayed(remote.prepare_server)(host) for host in hosts)

    main.ensure_results_dir()
    deterministic_cuda.ensure_results_dir()
    placement.ensure_results_dir()
    profiler.ensure_results_dir()

    with parallel_backend("loky", n_jobs=len(hosts)):
        Parallel()(
            delayed(experiment_per_host)(host, run_cpu, run_gpu) for host in hosts
        )
