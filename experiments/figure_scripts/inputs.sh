#!/bin/bash

python util/dendrogram.py latex results/input_distribution/cifar10_medium-conv2d_11/original/*json > ~/Projects/forennsic_paper/generated/input_distribution/original.tex
python util/dendrogram.py latex --no-labels results/input_distribution/cifar10_medium-conv2d_11/permutation_i1/*json > ~/Projects/forennsic_paper/generated/input_distribution/permutation_i0.tex
python util/dendrogram.py latex --no-labels results/input_distribution/cifar10_medium-conv2d_11/permutation_i0/*json > ~/Projects/forennsic_paper/generated/input_distribution/permutation_i1.tex
python util/dendrogram.py latex --no-labels results/input_distribution/cifar10_medium-conv2d_11/random/*json > ~/Projects/forennsic_paper/generated/input_distribution/random.tex
