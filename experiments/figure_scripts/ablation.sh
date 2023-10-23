#!/bin/bash

base=results/num_multiplications/single_layer/2022_09_01
for d in $base/conv_f*; do
  name=$(basename $d)
  echo $name
  python util/dendrogram.py pyplot $d/*json -o $base/dendrograms/$name
done
