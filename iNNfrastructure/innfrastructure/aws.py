import logging
from typing import Iterable, Dict, List, Union

import fabric
import invoke
from dataclasses import dataclass
import json

from invoke import UnexpectedExit

logger = logging.getLogger(__name__)

AWS_IMAGE_ID = "ami-0acb218a9a0302218"


@dataclass
class AwsInstanceConfiguration:
    """
    Container class for information required by
    https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html
    """

    name: str  # set as tag, as actual hostname setting isn't really possible
    instance_type: str
    spot_instance: bool = False
    image_id: str = AWS_IMAGE_ID
    key_name: str = "aws"
    security_group: str = "open"

    def get_command(self):
        command = " ".join(
            [
                "aws ec2 run-instances",
                f"--image-id {self.image_id}",
                f"--instance-type {self.instance_type}",
                f"--key {self.key_name}",
                f"--security-group-ids {self.security_group}",
                f"--tag-specifications 'ResourceType=instance,Tags=[{{Key=name,Value={self.name}}}]'",
                f"--iam-instance-profile Name=ecr_access",
            ]
        )
        return command


_CONFIGS = [
    AwsInstanceConfiguration("t3.large", "t3.large"),
    AwsInstanceConfiguration("t2.medium", "t2.medium"),
    AwsInstanceConfiguration("t2.large", "t2.large"),
    AwsInstanceConfiguration("t2.xlarge", "t2.xlarge"),
    AwsInstanceConfiguration("m5.large", "m5.large"),
    AwsInstanceConfiguration("r5.large", "r5.large"),
    AwsInstanceConfiguration("m4.large", "m4.large"),
    AwsInstanceConfiguration("c4.xlarge", "c4.xlarge"),
    AwsInstanceConfiguration("c5.xlarge", "c5.xlarge"),
    AwsInstanceConfiguration("c5n.xlarge", "c5n.xlarge"),
    AwsInstanceConfiguration("x1e.xlarge", "x1e.xlarge"),
    AwsInstanceConfiguration("z1d.large", "z1d.large"),
    AwsInstanceConfiguration("i3en.large", "i3en.large"),
    AwsInstanceConfiguration("c5.12xlarge", "c5.12xlarge"),
    AwsInstanceConfiguration("p2.xlarge", "p2.xlarge"),
    AwsInstanceConfiguration("g4dn.xlarge", "g4dn.xlarge"),
    AwsInstanceConfiguration("g3s.xlarge", "g3s.xlarge"),
    AwsInstanceConfiguration("p3.2xlarge", "p3.2xlarge"),
    AwsInstanceConfiguration("p3dn.24xlarge", "p3dn.24xlarge"),
    AwsInstanceConfiguration("p4d.24xlarge", "p4d.24xlarge"),
]
# noinspection DuplicatedCode
AWS_CONFIGS: Dict[str, AwsInstanceConfiguration] = {
    config.name: config for config in _CONFIGS
}


def create_instance(instance_config: AwsInstanceConfiguration):
    command = instance_config.get_command()
    invoke.run(command)


@dataclass
class AwsInstanceInfo:
    """Contains information about an existing AWS instance"""

    id: str
    ip: str
    name: str
    type: str


def get_running_instances():
    cmd = "aws ec2 describe-instances --filters=Name=instance-state-name,Values=running"
    raw_result = invoke.run(cmd, hide=True).stdout
    result = json.loads(raw_result)

    instance_infos = []

    for reservation in result["Reservations"]:
        for instance in reservation["Instances"]:
            ip = instance["NetworkInterfaces"][0]["Association"]["PublicIp"]
            instance_type = instance["InstanceType"]
            id = instance["InstanceId"]

            name = None
            try:
                for tag in instance["Tags"]:
                    if tag["Key"] == "name":
                        name = tag["Value"]
                        break
            except KeyError:
                logger.warning(
                    f"Untagged instance of type {instance_type} found at IP {ip}"
                )

            instance_infos.append(AwsInstanceInfo(id, ip, name, instance_type))

    return instance_infos


def delete_instance(instance: AwsInstanceInfo):
    cmd = f"aws ec2 terminate-instances --instance-ids {instance.id}"
    invoke.run(cmd, hide=True)


def delete_instances(instances: [AwsInstanceInfo]):
    cmd = f"aws ec2 terminate-instances --instance-ids "
    cmd += " " + " ".join([instance.id for instance in instances])
    invoke.run(cmd)


def ensure_configs_running(config_names: Iterable[str]):
    """Ensure that cloud configurations with the given names are running."""
    instances = get_running_instances()
    instance_names = [instance.name for instance in instances]
    missing_configs = [
        config_name for config_name in config_names if config_name not in instance_names
    ]

    for missing_config in missing_configs:
        command = AWS_CONFIGS[missing_config].get_command()
        invoke.run(command)


def cleanup_configs(config_names: List[str]):
    instances = [
        instance
        for instance in get_running_instances()
        if instance.name in config_names
    ]
    delete_instances(instances)


DOCKER_ARGUMENTS = [
    "-d",
    "--name forennsic",
    "-v /tmp:/tmp",
    "-it",
    "457863446379.dkr.ecr.eu-central-1.amazonaws.com/forennsic:latest",
]

DOCKER_GPU_ARGUMENTS = ["--gpus all"]


class AwsConnection(fabric.Connection):
    """Wraps fabric connection to allow for dockerized commands."""

    def __init__(
        self,
        host,
        user="ec2-user",
        port=None,
        config=None,
        gateway=None,
        forward_agent=None,
        connect_timeout=None,
        connect_kwargs=None,
        inline_ssh_env=None,
        setup=True,
    ):
        if connect_kwargs is None:
            connect_kwargs = {}
        if "key_filename" not in connect_kwargs:
            connect_kwargs["key_filename"] = "/home/alex/.ssh/aws.pem"
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
            self.run("sudo yum install -y amazon-ecr-credential-helper")
            self.run("mkdir -p ~/.docker")
            self.run(
                'echo \'{"credHelpers": {"457863446379.dkr.ecr.eu-central-1.amazonaws.com": "ecr-login"}}\' '
                "> ~/.docker/config.json"
            )
        try:
            self.run("docker container rm -f forennsic")
        except UnexpectedExit:
            pass

        docker_arguments = []
        try:
            self.run("ls /dev/nvidia0")
            docker_arguments += DOCKER_GPU_ARGUMENTS
        except UnexpectedExit:
            pass
        finally:
            docker_arguments += DOCKER_ARGUMENTS

        docker_command = "docker run " + " ".join(docker_arguments)
        self.run(docker_command)

    @fabric.connection.opens
    def docker_run(self, command: str, **kwargs) -> fabric.Result:
        dockerized_command = "docker exec forennsic " + command
        return self.run(dockerized_command, **kwargs)

    def __exit__(self, *exc):
        self.run("docker container rm -f forennsic")
        super().__exit__(*exc)
