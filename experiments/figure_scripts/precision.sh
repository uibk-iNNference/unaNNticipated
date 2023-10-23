#!/bin/bash

python util/dendrogram.py latex --calculate results/precision/cifar10_medium_float16/*json > ~/Projects/forennsic_paper_2.0/generated/precision/float16.tex
python util/dendrogram.py latex --calculate results/precision/cifar10_medium_float32/*json > ~/Projects/forennsic_paper_2.0/generated/precision/float32.tex
python util/dendrogram.py latex --calculate results/precision/cifar10_medium_float64/*json > ~/Projects/forennsic_paper_2.0/generated/precision/float64.tex

python util/dendrogram.py latex --calculate results/precision/cifar10_small_float16/*.json > ~/Projects/forennsic_paper_2.0/generated/precision/cifar10_small_float16.tex
python util/dendrogram.py latex --calculate results/precision/cifar10_small_float32/*.json > ~/Projects/forennsic_paper_2.0/generated/precision/cifar10_small_float32.tex
python util/dendrogram.py latex --calculate results/precision/cifar10_small_float64/*.json > ~/Projects/forennsic_paper_2.0/generated/precision/cifar10_small_float64.tex

python util/dendrogram.py latex --calculate results/precision/deep_weeds_float16/*.json > ~/Projects/forennsic_paper_2.0/generated/precision/deep_weeds_float16.tex
python util/dendrogram.py latex --calculate results/precision/deep_weeds_float32/*.json > ~/Projects/forennsic_paper_2.0/generated/precision/deep_weeds_float32.tex
python util/dendrogram.py latex --calculate results/precision/deep_weeds_float64/*.json > ~/Projects/forennsic_paper_2.0/generated/precision/deep_weeds_float64.tex
