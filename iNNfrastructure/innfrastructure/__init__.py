import fabric
from invoke import UnexpectedExit


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


class InnfrastructureConnection(fabric.Connection):
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
        docker_image="docker.uibk.ac.at:443/c7031199/forennsic",
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
