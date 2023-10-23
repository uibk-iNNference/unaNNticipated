#!/bin/bash

python winning_algorithm/eval.py indeterministic-float > ~/Projects/paper/generated/gpu_indeterministic.tex
python winning_algorithm/eval.py deterministic-float > ~/Projects/paper/generated/gpu_deterministic.tex
