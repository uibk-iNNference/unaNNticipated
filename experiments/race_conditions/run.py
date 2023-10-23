from innfrastructure import gcloud
import numpy as np
import json
from joblib import parallel_backend, Parallel, delayed
from posixpath import join
import os

CONFIGS = [
    "intel-broadwell-2",
    "intel-broadwell-4",
    "amd-rome-2",
    "amd-rome-4",
    "amd-rome-8",
    "amd-rome-16",
]

RESULT_DIR = join("results","race_conditions")

def experiment_per_host(instance):
    conn = gcloud.GcloudConnection(instance.ip)
    conn.put("/tmp/sample.npy","/tmp/sample.npy")

    results = [json.loads(conn.docker_run("innfcli predict cifar10_medium /tmp/sample.npy", hide=True).stdout) for _ in range(10)]

    all_predictions = [r['prediction'] for r in results]
    result = results[-1]
    result['all_predictions'] = all_predictions

    path = join(RESULT_DIR, instance.name + ".json")
    with open(path, 'w') as output_file:
        json.dump(result, output_file)

if __name__ == "__main__":
    gcloud.ensure_configs_running(CONFIGS)

    samples = np.load("data/datasets/cifar10/samples.npy")
    sample = samples[:1]
    np.save("/tmp/sample.npy",sample)

    os.makedirs(RESULT_DIR, exist_ok=True)

    instances = [instance for instance in gcloud.get_instances() if instance.name in CONFIGS]
    assert len(instances) == len(CONFIGS)
    with parallel_backend("loky", n_jobs=len(instances)):
        Parallel()(delayed(experiment_per_host)(instance) for instance in instances)

    gcloud.cleanup_configs(CONFIGS)
