import json
from typing import List, Union, Dict, Iterable
from dataclasses import dataclass
from copy import deepcopy

import fabric
import invoke
from invoke import UnexpectedExit
import innfrastructure

DEFAULT_ARGUMENTS = {
    "service_account": "forennsic@forennsic.iam.gserviceaccount.com",
    "boot_disk_size": "40GB",
    "image": "projects/cos-cloud/global/images/cos-85-13310-1453-16",
    "maintenance_policy": "TERMINATE",
}
GCP_GPU_STARTUP = r"""#! /bin/bash
sudo cos-extensions install gpu
sudo mount --bind /var/lib/nvidia /var/lib/nvidia
sudo mount -o remount,exec /var/lib/nvidia"""


def _add_default_arguments(current_args: Dict):
    for key, value in DEFAULT_ARGUMENTS.items():
        if key not in current_args:
            current_args[key] = value


class GcloudInstanceConfig:
    """Contains information required for creating a gcloud instance"""
    def __init__(self, name: str, zone: str, **kwargs):
        self.name = name
        self.zone = zone

        _add_default_arguments(kwargs)

        if "accelerator" in kwargs:
            try:
                kwargs["metadata"]["startup-script"] = GCP_GPU_STARTUP
            except KeyError:
                kwargs["metadata"] = {"startup-script": GCP_GPU_STARTUP}
                # TODO: clean up and fix

        self.additional_arguments = kwargs

    def add_arguments(self, additional_arguments: dict):
        for key, value in additional_arguments.items():
            self.additional_arguments[key] = value

    def get_command(self):
        command_parts = [
            "gcloud",
            "compute",
            "instances",
            "create",
            self.name,
            f"--zone={self.zone}",
        ]

        for key, value in self.additional_arguments.items():
            # gcloud uses dashes instead of underscores
            command_parts.append("--" + key.replace("_", "-"))

            if isinstance(value, list):
                value = ",".join(map(lambda x: str(x), value))

            if value is not None:
                string_value = rf"'{value}'"
                if isinstance(value, dict):
                    string_value = ""
                    for k, v in value.items():
                        string_value += (
                            "^:^" + k + "='" + v + "'"
                        )  # HACK: this is SUPER unsafe

                command_parts.append(string_value)

        s = " ".join(command_parts)
        return s


_CONFIGS: List[GcloudInstanceConfig] = [
    GcloudInstanceConfig(
        "intel-sandy-bridge",
        "europe-west4-a",
        machine_type="n1-standard-2",
        min_cpu_platform="Intel Sandy Bridge",
    ),
    GcloudInstanceConfig(
        "intel-ivy-bridge",
        "europe-west4-a",
        machine_type="n1-standard-2",
        min_cpu_platform="Intel Ivy Bridge",
    ),
    GcloudInstanceConfig(
        "intel-haswell",
        "europe-west4-a",
        machine_type="n1-standard-2",
        min_cpu_platform="Intel Haswell",
    ),
    GcloudInstanceConfig(
        "intel-broadwell",
        "europe-west4-a",
        machine_type="n1-standard-2",
        min_cpu_platform="Intel Broadwell",
    ),
    GcloudInstanceConfig(
        "intel-skylake",
        "europe-west4-a",
        machine_type="n1-standard-2",
        min_cpu_platform="Intel Skylake",
    ),
    GcloudInstanceConfig(
        "intel-ice-lake",
        "europe-west4-a",
        machine_type="n2-standard-2",
        min_cpu_platform="Intel Ice Lake",
    ),
    GcloudInstanceConfig(
        "intel-cascade-lake",
        "europe-west4-a",
        machine_type="n2-standard-2",
        min_cpu_platform="Intel Cascade Lake",
    ),
    GcloudInstanceConfig(
        "amd-rome",
        "europe-west4-a",
        machine_type="n2d-standard-2",
        min_cpu_platform="AMD Rome",
    ),
    GcloudInstanceConfig(
        "amd-milan",
        "europe-west4-a",
        machine_type="n2d-standard-2",
        min_cpu_platform="AMD Milan",
    ),
    # GPU
    GcloudInstanceConfig(
        "nvidia-t4",
        "europe-west1-b",
        machine_type="n1-standard-2",
        accelerator="count=1,type=nvidia-tesla-t4",
    ),
    GcloudInstanceConfig(
        "nvidia-k80",
        "europe-west1-b",
        machine_type="n1-standard-2",
        accelerator="count=1,type=nvidia-tesla-k80",
    ),
    GcloudInstanceConfig(
        "nvidia-v100",
        "europe-west4-a",
        machine_type="n1-standard-2",
        accelerator="count=1,type=nvidia-tesla-v100",
    ),
    GcloudInstanceConfig(
        "nvidia-p100",
        "europe-west4-a",
        machine_type="n1-standard-2",
        accelerator="count=1,type=nvidia-tesla-p100",
    ),
    GcloudInstanceConfig(
        "nvidia-a100",
        "europe-west4-a",
        machine_type="a2-highgpu-1g",
        accelerator="count=1,type=nvidia-tesla-a100",
    ),
]
GCP_CONFIGS: Dict[str, GcloudInstanceConfig] = {
    config.name: config for config in _CONFIGS
}


def create_instance(instance_config: GcloudInstanceConfig):
    command = instance_config.get_command()

    invoke.run(command)


@dataclass
class GcloudInstanceInfo:
    """Contains information about an existing GCloud instance"""
    name: str
    ip: str
    zone: str
    status: str


def get_instances() -> List[GcloudInstanceInfo]:
    command = ["gcloud compute instances list", "--format json"]
    output = invoke.run(
        " ".join(command),
        hide=True,
    ).stdout

    parsed = json.loads(output)

    instance_infos = []
    for current_dict in parsed:
        name = current_dict["name"]
        zone = current_dict["zone"].split("/")[-1]
        status = current_dict["status"]
        ip = (
            current_dict["networkInterfaces"][0]["accessConfigs"][0]["natIP"]
            if status == "RUNNING"
            else None
        )

        instance_info = GcloudInstanceInfo(name, ip, zone, status)
        instance_infos.append(instance_info)

    return instance_infos


def stop_instance(instance: Union[GcloudInstanceConfig, GcloudInstanceInfo]) -> None:
    command = [
        "gcloud",
        "compute",
        "instances",
        "stop",
        "-q",
        instance.name,
        "--zone",
        instance.zone,
    ]

    invoke.run(" ".join(command))


def delete_instance(instance_config: Union[GcloudInstanceConfig, GcloudInstanceInfo]):
    command = [
        "gcloud",
        "compute",
        "instances",
        "delete",
        "-q",
        instance_config.name,
        "--zone",
        instance_config.zone,
    ]

    invoke.run(" ".join(command), hide=True)


def ensure_configs_running(config_names: Iterable[str]):
    """Ensure that cloud configurations with the given names are running."""
    instances = get_instances()
    instance_names = [instance.name for instance in instances]
    missing_configs = [
        config_name for config_name in config_names if config_name not in instance_names
    ]

    for missing_config in missing_configs:
        try:
            invoke.run(GCP_CONFIGS[missing_config].get_command())
        except KeyError:
            assert (
                "nvidia" not in missing_config.lower()
            )  # the following only works for CPUs
            config_parts = missing_config.split("-")
            core_count = int(config_parts[-1])
            best_match = next(
                filter(lambda s: "-".join(config_parts[:-1]) in s, GCP_CONFIGS.keys())
            )
            print(f"Using {best_match} as base configuration for {missing_config}")

            adjusted_config = deepcopy(GCP_CONFIGS[best_match])
            adjusted_config.name += f"-{core_count}"
            machine_type = adjusted_config.additional_arguments["machine_type"]
            machine_type = machine_type.split("-")[:-1] + [str(core_count)]
            adjusted_config.additional_arguments["machine_type"] = "-".join(
                machine_type
            )

            invoke.run(adjusted_config.get_command())


def cleanup_configs(config_names: List[str]):
    for instance in get_instances():
        if instance.name not in config_names:
            continue

        delete_instance(instance)


DOCKER_ARGUMENTS = [
    "-d",
    "--name forennsic",
    "-v /tmp:/tmp",
    "-it",
]
DOCKER_GPU_ARGUMENTS = [
    "--volume /var/lib/nvidia/lib64:/usr/local/nvidia/lib64",
    "--volume /var/lib/nvidia/bin:/usr/local/nvidia/bin",
    "--device /dev/nvidia0:/dev/nvidia0",
    "--device /dev/nvidia-uvm:/dev/nvidia-uvm",
    "--device /dev/nvidiactl:/dev/nvidiactl",
]


class GcloudConnection(fabric.Connection):
    """Wraps fabric connection to allow for dockerized commands"""
    def __init__(
        self,
        host,
        user=None,
        port=None,
        config=None,
        gateway=None,
        forward_agent=None,
        connect_timeout=None,
        connect_kwargs=None,
        inline_ssh_env=None,
        docker_image="eu.gcr.io/forennsic/experiments",
        setup=True,
    ):
        super().__init__(
            host,
            user,
            port,
            config,
            gateway,
            forward_agent,
            connect_timeout,
            connect_kwargs,
            inline_ssh_env,
        )

        if setup:
            self.run("docker-credential-gcr configure-docker")
            print("Set up cloud credentials")
        try:
            self.run("docker container rm -f forennsic")
        except UnexpectedExit:
            pass

        docker_arguments = []
        try:
            self.run("ls /dev/nvidia0", hide=True)
            docker_arguments += DOCKER_GPU_ARGUMENTS
        except UnexpectedExit:
            pass
        finally:
            docker_arguments += DOCKER_ARGUMENTS + [docker_image]

        docker_command = "docker run " + " ".join(docker_arguments)
        self.run(docker_command)

    @fabric.connection.opens
    def docker_run(self, command: str, **kwargs) -> fabric.Result:
        dockerized_command = "docker exec forennsic " + command
        return self.run(dockerized_command, **kwargs)

    def __exit__(self, *exc):
        self.run("docker container rm -f forennsic")
        super().__exit__(*exc)
