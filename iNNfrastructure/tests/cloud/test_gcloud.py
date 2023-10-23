import copy
import pytest

from invoke import UnexpectedExit

from innfrastructure import gcloud, config


@pytest.fixture
def instance_config(dirname):
    return gcloud.GcloudInstanceConfig(
        "innfrastructure-test", "europe-west1-b",
        machine_type="e2-micro"
    )


def test_gcloud_instance_config(dirname):
    instance_config = gcloud.GcloudInstanceConfig(
        "config-test", "zone", additional="argument")
    instance_config.add_arguments({"second": "argument"})

    assert instance_config.additional_arguments["additional"] == "argument"
    assert instance_config.additional_arguments["second"] == "argument"

    assert instance_config.additional_arguments["additional"] == "argument"
    assert instance_config.additional_arguments["second"] == "argument"


# def test_gcloud_start_stop(instance_config):
#     # subprocess.run raises an Error if problems occur
#     try:
#         gcloud.create_instance(instance_config)
#         gcloud.delete_instance(instance_config)
#     except CalledProcessError as e:
#         raise AssertionError(
#             f"""Gcloud command failed with stderr:
# {e.stderr}
# stdout:
# {e.stdout}""")


def test_gcloud_failure():
    instance_config = gcloud.GcloudInstanceConfig(
        "failure-test", "europe-west1-b",
        breaking="this is not an argument lol",
        list_param=list(range(5))
    )

    with pytest.raises(UnexpectedExit):
        gcloud.create_instance(instance_config)


def test_gcloud_get_instances(instance_config: gcloud.GcloudInstanceConfig):
    # gcloud.create_instance(instance_config)
    instance_infos = gcloud.get_instances()

    found = False
    print(instance_infos)
    for instance_info in instance_infos:
        if instance_info.name == instance_config.name:
            found = True
            break

    if not found:
        raise AssertionError("Did not find the created instance")

    gcloud.delete_instance(instance_config)
