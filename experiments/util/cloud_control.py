from dataclasses import dataclass
from innfrastructure import gcloud, aws
import click
import invoke


@dataclass
class GCPConfig:
    zone: str
    machine_type: str
    accelerator: str = None
    min_cpu_platform: str = None

    def get_name(self) -> str:
        try:
            if self.accelerator is not None:
                return self.accelerator

            if (
                "broadwell" in self.min_cpu_platform.lower()
                or "rome" in self.min_cpu_platform.lower()
            ):
                return (
                    self.min_cpu_platform.replace(" ", "-").lower()
                    + "-"
                    + self.machine_type.lower()
                )

            if self.min_cpu_platform is not None:
                return self.min_cpu_platform.replace(" ", "-").lower()
        except AttributeError:
            return "amd-milan"  # this is bound to break things somewhen


GCP_CPU_CONFIGS = [
    GCPConfig("europe-west4-a", "n1-standard-2", min_cpu_platform="Intel Sandy Bridge"),
    GCPConfig("europe-west4-a", "n1-standard-2", min_cpu_platform="Intel Ivy Bridge"),
    GCPConfig("europe-west4-a", "n1-standard-2", min_cpu_platform="Intel Haswell"),
    GCPConfig("europe-west4-a", "n1-standard-2", min_cpu_platform="Intel Broadwell"),
    GCPConfig("europe-west4-a", "n1-standard-4", min_cpu_platform="Intel Broadwell"),
    GCPConfig("europe-west4-a", "n1-standard-2", min_cpu_platform="Intel Skylake"),
    GCPConfig("europe-west4-a", "n2-standard-2", min_cpu_platform="Intel Ice Lake"),
    GCPConfig("europe-west4-a", "n2-standard-2", min_cpu_platform="Intel Cascade Lake"),
    GCPConfig("europe-west4-a", "n2d-standard-2", min_cpu_platform="AMD Rome"),
    GCPConfig("europe-west4-a", "n2d-standard-4", min_cpu_platform="AMD Rome"),
    GCPConfig("europe-west4-a", "n2d-standard-8", min_cpu_platform="AMD Rome"),
    GCPConfig("europe-west4-a", "n2d-standard-16", min_cpu_platform="AMD Rome"),
    GCPConfig("europe-west4-a", "n2d-standard-32", min_cpu_platform="AMD Rome"),
    GCPConfig("europe-west4-a", "n2d-standard-48", min_cpu_platform="AMD Rome"),
    GCPConfig("europe-west4-a", "n2d-standard-2", min_cpu_platform="AMD Milan"),
]

GCP_GPU_CONFIGS = [
    GCPConfig("europe-west1-b", "n1-standard-2", accelerator="nvidia-tesla-t4"),
    GCPConfig("europe-west1-b", "n1-standard-2", accelerator="nvidia-tesla-k80"),
    GCPConfig("europe-west1-b", "n1-standard-8", accelerator="nvidia-tesla-k80"),
    GCPConfig("europe-west4-a", "n1-standard-2", accelerator="nvidia-tesla-v100"),
    GCPConfig("europe-west4-a", "n1-standard-2", accelerator="nvidia-tesla-p100"),
    GCPConfig("europe-west4-a", "a2-highgpu-1g", accelerator="nvidia-tesla-a100"),
]


def create_instance(gcp_config: GCPConfig) -> None:
    instance_config = gcloud.GcloudInstanceConfig(
        gcp_config.get_name(),
        zone=gcp_config.zone,
        machine_type=gcp_config.machine_type,
        image="projects/cos-cloud/global/images/cos-85-13310-1453-16",
        boot_disk_size="20GB",
        service_account="forennsic@forennsic.iam.gserviceaccount.com",
        preemptible=None,
    )
    if gcp_config.accelerator is not None:
        instance_config.add_arguments(
            {"accelerator": f"count=1,type={gcp_config.accelerator}"}
        )
    if gcp_config.min_cpu_platform is not None:
        instance_config.add_arguments({"min_cpu_platform": gcp_config.min_cpu_platform})

    print(f"Launching instance {instance_config.name}...")
    gcloud.create_instance(instance_config)


@click.group()
def main():
    pass


@main.group()
def gcp():
    pass


@gcp.command("create-gpu")
def gcp_create_gpu_instances():
    for gpu_config in GCP_GPU_CONFIGS:
        create_instance(gpu_config)


@gcp.command("create-cpu")
def gcp_create_cpu_instances():
    for cpu_config in GCP_CPU_CONFIGS:
        create_instance(cpu_config)


@gcp.command("base")
def gcp_management_instance():
    instance_config = gcloud.GcloudInstanceConfig(
        name="base",
        zone="europe-west4-a",
        machine_type="n1-standard-2",
        image="projects/ubuntu-os-cloud/global/images/ubuntu-2004-focal-v20211212",
        boot_disk_size="30GB",
        service_account="forennsic@forennsic.iam.gserviceaccount.com",
        preemptible=None,
    )
    print("Launching setup instance...")
    gcloud.create_instance(instance_config)


AWS_CPU_CONFIGS = [
    "t3.large",
    "t2.medium",
    "t2.large",
    "t2.xlarge",
    "m5.large",
    "r5.large",
    "m4.large",
    "c4.xlarge",
    "c5.xlarge",
    "c5.4xlarge",
    "c5n.xlarge",
    "x1e.xlarge",
    "z1d.large",
    "i3en.large",
    "c5.12xlarge",
]
AWS_GPU_CONFIGS = [
    # "p2.xlarge",
    # "g4dn.xlarge",
    "g3s.xlarge",
    # "p3.2xlarge",
    # "p3dn.24xlarge",
    # "p4d.24xlarge",
]
AWS_IMAGE_ID = "ami-04a584ac67924a036"
AWS_SNAPSHOT_ID = "snap-0465fce9cea1805da"


@main.group("aws")
def aws_commands():
    pass


@aws_commands.command("base")
def aws_management_instance():
    hostname = "management"

    instance_config = aws.AwsInstanceConfiguration(
        instance_type="t2.large",
        host_name=hostname,
        image_id=AWS_IMAGE_ID,
        key_name="aws",
        security_groups=["open"],
        spot_instance=False,
        block_device_mappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "DeleteOnTermination": False,
                    "SnapshotId": AWS_SNAPSHOT_ID,
                    "VolumeSize": 30,
                    "VolumeType": "gp3",
                },
            }
        ],
    )
    cmd = instance_config.get_command()
    print(f"Creating AWS management instance")
    invoke.run(cmd, hide=True)


def _create_aws_instance(instance_type, spot_instance):
    hostname = instance_type.replace(".", "-")

    instance_config = aws.AwsInstanceConfiguration(
        instance_type=instance_type,
        host_name=hostname,
        image_id=AWS_IMAGE_ID,
        key_name="aws",
        security_groups=["open"],
        spot_instance=spot_instance,
        block_device_mappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "SnapshotId": AWS_SNAPSHOT_ID,
                    "VolumeSize": 30,
                    "VolumeType": "gp3",
                },
            }
        ],
    )
    cmd = instance_config.get_command()
    print(f"Creating instance type {instance_type}...")
    invoke.run(cmd, hide=True)


@aws_commands.command("create-cpu")
def aws_create_cpu_instances():
    for instance_type in AWS_CPU_CONFIGS:
        _create_aws_instance(instance_type, spot_instance=True)


@aws_commands.command("create-gpu")
def aws_create_gpu_instances():
    for instance_type in AWS_GPU_CONFIGS:
        _create_aws_instance(instance_type, spot_instance=False)


if __name__ == "__main__":
    main()
