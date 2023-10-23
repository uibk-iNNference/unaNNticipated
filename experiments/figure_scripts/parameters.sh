#!/bin/bash

python util/dendrogram.py latex results/parameter_distribution/cifar10_medium-conv2d_15/extracted_layer-samples_normal_seed1337/*json > ~/Projects/forennsic_paper/generated/parameter_distribution/model_weights.tex
python util/dendrogram.py latex --no-labels results/parameter_distribution/cifar10_medium-conv2d_15/without_bias-samples_normal_seed1337/*json > ~/Projects/forennsic_paper/generated/parameter_distribution/without_bias.tex
python util/dendrogram.py latex --no-labels results/parameter_distribution/cifar10_medium-conv2d_15/fitted_distribution-samples_normal_seed1337/*json > ~/Projects/forennsic_paper/generated/parameter_distribution/fitted.tex
python util/dendrogram.py latex --no-labels results/parameter_distribution/cifar10_medium-conv2d_15/default_distribution-samples_normal_seed1337/*json > ~/Projects/forennsic_paper/generated/parameter_distribution/default.tex
