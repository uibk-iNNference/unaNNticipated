#!/bin/bash

python main/eval/dendrogram.py --base-dir results/secondary/conv_2d_parameters/combined/conv_f1_k4_i103x103x3 -m bits '' &
python main/eval/dendrogram.py --base-dir results/secondary/conv_2d_parameters/combined/conv_f3_k7_i34x34x3 -m bits '' &
python main/eval/dendrogram.py cifar10_medium -m bits &
