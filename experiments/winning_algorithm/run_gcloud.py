from innfrastructure import gcloud, remote
from common import run_on_all_hosts, RunConfiguration, HostConfiguration
from posixpath import join

RESULT_DIR = join("results", "winning_algorithm", "gcloud")

GPU_CONFIGS = [
    "nvidia-t4",
    "nvidia-k80",
    "nvidia-v100",
    "nvidia-p100",
    "nvidia-a100",
]


def extract_host_config(instance) -> HostConfiguration:
    return HostConfiguration(address=instance.ip, hostname=instance.name)


def generate_cifar_configs():
    return [
        RunConfiguration(
            f"cifar10_{size}",
            join("data", "datasets", "cifar10", "samples.npy"),
            [0, 1, 6],
            RESULT_DIR,
        )
        # for size in ["small", "medium"]
        for size in ["small"]
    ]


def main():
    # run_configurations = [
    #     RunConfiguration(
    #         "deep_weeds",
    #         join("data", "datasets", "deep_weeds", "samples.npy"),
    #         [0, 1, 6],
    #         RESULT_DIR,
    #     )
    # ] + generate_cifar_configs()
    run_configurations = generate_cifar_configs()

    gcloud.ensure_configs_running(GPU_CONFIGS)
    instances = [instance for instance in gcloud.get_instances() if instance.name in GPU_CONFIGS]
    assert len(instances) == len(GPU_CONFIGS)
    host_configs = [extract_host_config(instance) for instance in instances]

    for host_config in host_configs:
        remote.update_host_keys(host_config.address)

    for run_config in run_configurations:
        run_on_all_hosts(run_config, host_configs)

    gcloud.cleanup_configs(GPU_CONFIGS)

if __name__ == "__main__":
    main()
