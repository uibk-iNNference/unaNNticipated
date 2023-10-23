import streamlit as st

from main.eval import determinism_cpu, determinism_gpu, equivalence_classes
from secondary.nondeterminism_location import eval as nondeterminism_location
from secondary.deterministic_cuda import eval as deterministic_cuda

st.title("Main Experiments")
equivalence_classes.visualize()
determinism_gpu.visualize()
determinism_cpu.visualize()

st.title("Searching for Root Causes")
nondeterminism_location.visualize()
deterministic_cuda.visualize()
