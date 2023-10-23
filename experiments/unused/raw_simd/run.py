import os
import sys
import tempfile
from posixpath import join
from typing import Tuple, List

import invoke
import numpy as np
from fabric import Connection
from joblib import Parallel, delayed, parallel_backend

from innfrastructure import remote, gcloud

REMOTE_TEMPDIR = "/tmp/raw_simd"
RESULT_DIR = join('results', 'raw_simd', "gcloud")


def main(upload: bool):
    a, b = generate_inputs()

    # assert that inputs are seeded and consistent
    a2, b2 = generate_inputs()
    assert np.array_equal(a, a2)
    assert np.array_equal(b, b2)

    temp_dir = tempfile.TemporaryDirectory(prefix="forennsic_raw_simd")
    np.save(join(temp_dir.name, "a.npy"), a)
    np.save(join(temp_dir.name, "b.npy"), b)

    upload_command = f"gsutil -m cp -r ../innfrastructure {temp_dir.name}/*.npy gs://forennsic-conv2/raw_simd/"
    if upload:
        invoke.run(upload_command)

    os.makedirs(RESULT_DIR, exist_ok=True)
    instances: List[gcloud.GcloudInstanceInfo] = gcloud.get_instances()

    with parallel_backend('loky'):
        Parallel()(delayed(setup_machine)(instance.ip) for instance in instances)
        Parallel()(delayed(run_experiment)(instance) for instance in instances)


def generate_inputs() -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1337)  # seed for consistency

    a = rng.random((16, 1))
    b = rng.random((16, 1))
    return a, b


def setup_machine(host: str):
    remote.update_host_keys(host)
    # rsync innfrastructure
    with Connection(host) as connection:
        connection.run(f"gsutil -m cp -r gs://forennsic-conv2/raw_simd /tmp/")

        with connection.cd(f"{REMOTE_TEMPDIR}"):
            # create and source venv
            connection.run("python3 -m venv venv")
            with connection.prefix("source venv/bin/activate"):
                connection.run("cd innfrastructure && python setup.py clean")  # remove any cached cmake stuff
                connection.run("pip install -e innfrastructure")


def run_experiment(instance: gcloud.GcloudInstanceInfo):
    with Connection(instance.ip) as conn:
        with conn.cd(REMOTE_TEMPDIR):
            with conn.prefix("source venv/bin/activate"):
                remote_output_path = f"{REMOTE_TEMPDIR}/result.json"
                conn.run(
                    f"innfcli cpp simd {REMOTE_TEMPDIR}/a.npy {REMOTE_TEMPDIR}/b.npy" +
                    f" --output_path {remote_output_path}")

                target_name = f"{instance.name}.json"
                target_path = join(RESULT_DIR, target_name)
                conn.get(remote_output_path, target_path)


if __name__ == "__main__":
    upload = "--upload" in sys.argv
    main(upload)
