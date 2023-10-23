import itertools
import re
from typing import List, Union, Dict
import cpuinfo
import socket
import psutil

import base64
import numpy as np
from invoke import run, UnexpectedExit

get_cpu_info = cpuinfo.get_cpu_info
get_hostname = socket.gethostname


def convert_np_to_json(array: np.ndarray) -> dict:
    """Convert numpy ndarray to base64 encoded buffer, retain dtype and shape in json dict"""
    if array is None:
        return None

    array_bytes = array.tobytes()
    encoded_bytes = base64.standard_b64encode(array_bytes)
    decoded_bytes = encoded_bytes.decode("UTF-8")
    ret = {"bytes": decoded_bytes, "dtype": str(array.dtype), "shape": array.shape}
    return ret


def convert_json_to_np(encoded_array: dict) -> np.ndarray:
    """Convert json dict to numpy ndarray, see convert_np_to_jsons"""
    if encoded_array is None:
        return None
    byte_string = encoded_array["bytes"]
    dtype = encoded_array["dtype"]
    shape = encoded_array["shape"]

    array_string = byte_string.encode("UTF-8")
    array_bytes = base64.standard_b64decode(array_string)
    array = np.frombuffer(array_bytes, dtype=dtype)
    return array.reshape(shape)


def get_devices() -> List[str]:
    from tensorflow.python.client.device_lib import list_local_devices

    return [x.name for x in list_local_devices()]


def get_device_info(device: Union[str, None]) -> Union[None, Dict[str, str]]:
    """Get additional information about a device. Returns None if device is None or not found."""
    if device is None:
        return None

    # import in here to reduce module load time
    from tensorflow.python.client.device_lib import list_local_devices

    entries = list_local_devices()

    # search for correct device
    target_entry = None
    for entry in entries:
        if entry.name.lower() == device.lower():
            target_entry = entry
            break

    if target_entry is not None:
        # convert to dict
        ret = {
            "name": target_entry.name,
            "device_type": target_entry.device_type,
            "memory_limit": target_entry.memory_limit,
            "incarnation": target_entry.incarnation,
        }

        physical_desc = target_entry.physical_device_desc
        if physical_desc:  # pragma: no cover
            # split physical description
            desc_entries = physical_desc.split(", ")
            desc_dict = dict([entry.split(": ") for entry in desc_entries])
            ret["physical_description"] = desc_dict

        return ret

    return None


def get_commit() -> str:
    result = run("git rev-parse HEAD", hide=True)
    return result.stdout.strip()


def clean_gpu_name(gpu_name):
    brand = "NVIDIA "
    if gpu_name.startswith(brand):
        gpu_name = gpu_name[len(brand) :]

    partial_prefix = "GeForce "
    if gpu_name.startswith(partial_prefix):
        gpu_name = gpu_name[len(partial_prefix) :]

    return gpu_name


def clean_cpu_name(cpu_name):
    # split off Intel clock speed
    cpu_name = cpu_name.split("@")[0]

    # drop (R)
    cpu_name = cpu_name.replace("(R)", "")

    # drop CPU
    cpu_name = cpu_name.replace("CPU", "")

    # drop Core(TM)
    cpu_name = cpu_name.replace("Core(TM)", "")

    return cpu_name


def get_clean_device_name(result):
    try:
        return clean_gpu_name(result["device"]["physical_description"]["name"]).replace(
            " ", "_"
        )
    except KeyError:
        return clean_cpu_name(result["cpu_info"]["brand_raw"]).replace(" ", "_")


def get_stack_info() -> Union[None, Dict[str, Union[str, None]]]:
    """Collect information about NVidia CUDA and driver version."""
    # get information from nvidia-smi
    result = run("nvidia-smi -q", hide=True, warn=True)
    if result.return_code != 0:
        return None

    driver_version = None
    cuda_version = None

    for line in result.stdout.splitlines():
        if line.startswith("Driver Version"):
            driver_version = line.split(":")[1].strip()
            continue

        if line.startswith("CUDA Version"):
            cuda_version = line.split(":")[1].strip()

    return {"driver_version": driver_version, "cuda_version": cuda_version}


def get_memory():
    return psutil.virtual_memory()._asdict()


RELEVANT_FLAGS = {
    # "adx", # intel arbitrary precision arithmetic
    "aes",
    "avx2",
    "avx",
    "avx512f",
    "avx512cd",
    "avx512pf",
    "avx512er",
    "avx512vl",
    "avx512bw",
    "avx512dq",
    "avx512vbmi",
    "avx512ifma",
    "avx512_4vnniw",
    "avx512_4fmaps",
    "bmi1",
    "bmi2",
    "cmov",
    "cmpxchg16b",
    "cmpxchg8b",
    # "f16c", # 16 bit float conversion instructions (not used for our measurements, but interesting)
    "fma",
    "mmx",
    "pclmulqdq",
    "popcnt",
    "prefetchw",
    "prefetchwt1",
    # "rdrand", # on chip hardware PRNG
    # "rdseed", # access to the entropy generating hardware for rdrand
    # "smap", # blocks ring 0 execution of user accessible code
    "sse2",
    "sse3",
    "sse4_1",
    "sse4_2",
    "sse",
    "ssse3",
    "hypervisor",
    # up to here they are taken from tensorflow source code at
    # tensorflow/core/platform/cpu_info.cc, for release tag 2.5
    # the flag below has been found through manual analysis
    "misalignsse",
}

CORE_SPLITS = [
    0,
    4,
    8,
    16,
    32,
]


def extract_cpu_stats(result: Dict):
    """Extract relevant CPU stats from cpuinfo result.
    Includes subset of flags and binned core count."""
    if result["device_type"] == "gpu":
        return result["device_name"]

    # gb = round(result['memory']['total'] / 10 ** 9)
    core_count = result["cpu_info"]["count"]
    core_split = None
    for i, split in enumerate(CORE_SPLITS):
        if core_count >= split:
            core_split = i
        else:
            break

    flags = frozenset(result["cpu_info"]["flags"])
    filtered_flags = flags & RELEVANT_FLAGS
    cpu_stat = (filtered_flags, core_split)
    return cpu_stat


def get_winning_algorithm(operations: List, return_operation: bool = False) -> str:
    """Get the winning algorithm from a list of operations from TF profiler"""
    convolution_regexes = {
        re.compile(r"explicit_convolve_sgemm"): "explicit sgemm",
        re.compile(r"implicit_convolve_sgemm"): "implicit sgemm",
        re.compile(r"precomputed_convolve_sgemm"): "precomputed sgemm",
        re.compile(r"conv2d_grouped_direct_kernel"): "grouped naive kernel",
        re.compile(r"fft2d_c2r"): "FFT gemm",
        re.compile(r"cudnn_convolve_sgemm_sm35"): "Kepler implicit sgemm",
        re.compile(
            r"cudnn_convolve_precomputed_sgemm_sm35"
        ): "Kepler precomputed sgemm",
        re.compile(r"maxwell_scudnn_winograd"): "Maxwell Winograd",
        re.compile(r"maxwell_sgemm"): "Maxwell Winograd nonfused",
        re.compile(r"gemmSN_NN_kernel"): "Turing Winograd nonfused",
        re.compile(r"volta_scudnn_winograd"): "Volta compiled winograd",
        re.compile(r"volta_scudnn_[\dx]+_relu"): "Volta fused conv/ReLU",
        re.compile(r"volta_sgemm"): "Volta nonfused Winograd",
        re.compile(
            r"xmma_cudnn::gemm::kernel<xmma_cudnn::implicit_gemm"
        ): "Ampere implicit gemm",
        re.compile(r"xmma_fprop_implicit_gemm"): "Ampere fprop implicit gemm",
        re.compile(r"ampere_scudnn_winograd"): "Ampere Winograd",
    }

    best_match = None
    matching_operation = None

    for operation, (regex, alg_name) in itertools.product(
        operations, convolution_regexes.items()
    ):
        regex_match = regex.search(operation)
        if regex_match is not None:
            if best_match is not None and alg_name != best_match:
                raise ValueError(
                    f"Had previous best match {best_match}, new match {alg_name}, operations are:\n"
                    + "\n\n".join(operations)
                )

            best_match = alg_name
            matching_operation = operation

    if best_match is None:
        raise AssertionError(
            "could not find best match for conv, operations are:\n"
            + "\n\n".join(operations)
        )

    if return_operation:
        return best_match, matching_operation
    return best_match
