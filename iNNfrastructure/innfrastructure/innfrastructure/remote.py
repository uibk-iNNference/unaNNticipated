from time import sleep

import invoke
from invoke import UnexpectedExit
from paramiko.ssh_exception import NoValidConnectionsError

def update_host_keys(host: str):
    for i in range(10):
        try:
            # fix hostkeys
            invoke.run(f"ssh-keygen -R {host}")
            invoke.run(f"ssh-keyscan {host} >> ~/.ssh/known_hosts")

            break
        except (NoValidConnectionsError, UnexpectedExit):
            print(
                f"Could not connect to experiment instance {host} on try {i}, waiting and retrying..."
            )
            sleep(15)
            continue


