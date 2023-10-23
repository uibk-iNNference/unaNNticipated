#!/bin/bash

python instrumental/activation_tables.py precision
python instrumental/activation_tables.py deviations > ~/Projects/forennsic_paper_2.0/generated/deviation_counts.tex
