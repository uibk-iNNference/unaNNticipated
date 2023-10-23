from innfrastructure import metadata

import numpy as np


def test_random_array_encoding_decoding():
    array = np.random.random(10)

    assert np.all(
        array == metadata.convert_json_to_np(metadata.convert_np_to_json(array))
    )
    assert np.all(
        metadata.convert_json_to_np(metadata.convert_np_to_json(None)) is None
    )


def test_get_devices():
    name_list = metadata.get_devices()

    assert "/device:CPU:0" in name_list


def test_get_device_info():
    device_info = metadata.get_device_info("/device:CPU:0")

    assert device_info["name"] == "/device:CPU:0"
    assert device_info["device_type"] == "CPU"
    assert "memory_limit" in device_info
    assert "incarnation" in device_info

    # test for nonexisting device
    assert metadata.get_device_info("notfound") is None


def test_empty_device_info():
    assert metadata.get_device_info(None) is None


def test_stack_info():
    data = metadata.get_stack_info()
    if data is not None:
        assert 'driver_version' in data
        assert 'cuda_version' in data
