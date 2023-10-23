MODEL_TYPE_SORT_INDEX = {"cifar10_small": 0, "cifar10_medium": 1, "deep_weeds": 2}

AMD_MICROARCHITECTURES = {
    # information taken from https://en.wikichip.org/wiki/amd/cpuid
    (0x17, 49): "Rome",
    (0x19, 1): "Milan",
}

INTEL_MICROARCHITECTURES = {
    # information taken from https://en.wikichip.org/wiki/intel/cpuid
    45: "Sandy Br.",
    62: "Ivy Br.",
    63: "Haswell",
    79: "Broadwell",
    85: "Skylake",
    106: "Ice Lake",
    158: "Coffee Lake",
}

NVIDIA_MICROARCHITECTURES = {
    # information taken from https://www.techpowerup.com/gpu-specs
    "Tesla K80": "Kepler",
    "Tesla T4": "Turing",
    "Tesla M60": "Maxwell",
    "A100-SXM4-40GB": "Ampere",
    "Tesla P100-PCIE-16GB": "Pascal",
    "Tesla V100-SXM2-16GB": "Volta",
    "GTX 1650": "Turing",
    "GTX 970": "Maxwell",
    "GTX 980": "Maxwell",
    "RTX 2070": "Turing",
}

NVIDIA_SORTING = {
    "Kepler": 0,
    "Maxwell": 1,
    "Pascal": 2,
    "Volta": 3,
    "Turing": 4,
    "Ampere": 5,
}

# tensorflow/core/platform/cpu_info.cc, for release tag 2.5
RELEVANT_FLAGS = {
    "3dnowext",
    "abm",
    "avx2",
    "avx512_bitalg",
    "avx512_vbmi2",
    "avx512_vpopcntdq",
    "avx512bitalg",
    "avx512bw",
    "avx512cd",
    "avx512dq",
    "avx512f",
    "avx512ifma",
    "avx512vbmi",
    "avx512vbmi2",
    "avx512vl",
    "avx512vpopcntdq",
    "bmi1",
    "bmi2",
    "clzero",
    "cmp_legacy",
    "cr8_legacy",
    "extd_apicid",
    "fma",
    "fxsr_opt",
    "gfni",
    "misalignsse",
    "mmxext",
    "movbe",
    "npt",
    "nrip_save",
    "osvw",
    "rdpid",
    "sse4a",
    "topoext",
    "vaes",
    "vmmcall",
    "vpclmulqdq",
    "xsaveerptr",
    "xsaves",
}
