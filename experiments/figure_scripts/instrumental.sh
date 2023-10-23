#!/bin/bash

python instrumental/eval.py output cifar10_small > ~/Projects/forennsic_paper_2.0/generated/instrumental_small.dat
python instrumental/eval.py output cifar10_small_untrained > ~/Projects/forennsic_paper_2.0/generated/instrumental_small_untrained.dat
python instrumental/eval.py output cifar10_small_sigmoid > ~/Projects/forennsic_paper_2.0/generated/instrumental_small_sigmoid_trained.dat
python instrumental/eval.py output cifar10_small_linear > ~/Projects/forennsic_paper_2.0/generated/instrumental_small_linear_trained.dat
python instrumental/eval.py output cifar10_small_sigmoid_untrained > ~/Projects/forennsic_paper_2.0/generated/instrumental_small_sigmoid_untrained.dat

python instrumental/eval.py output cifar10_medium > ~/Projects/forennsic_paper_2.0/generated/instrumental_medium.dat
python instrumental/eval.py output cifar10_medium_untrained > ~/Projects/forennsic_paper_2.0/generated/instrumental_medium_untrained.dat
python instrumental/eval.py output cifar10_medium_sigmoid > ~/Projects/forennsic_paper_2.0/generated/instrumental_medium_sigmoid_trained.dat
python instrumental/eval.py output cifar10_medium_linear > ~/Projects/forennsic_paper_2.0/generated/instrumental_medium_linear_trained.dat
python instrumental/eval.py output cifar10_medium_sigmoid_untrained > ~/Projects/forennsic_paper_2.0/generated/instrumental_medium_sigmoid_untrained.dat
