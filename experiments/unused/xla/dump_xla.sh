#!/bin/bash

XLA_FLAGS="--xla_dump_to=results/monitoring" TF_XLA_FLAGS="--tf_xla_auto_jit=2" python minimal.py
