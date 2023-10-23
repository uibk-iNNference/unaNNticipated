#!/bin/bash

python util/dendrogram.py pyplot -m bits results/equivalence_classes/gcloud/cpu/*_32-*medium*0.json &
python util/dendrogram.py pyplot -m bits results/equivalence_classes/gcloud/cpu/*_16-*medium*0.json &
python util/dendrogram.py pyplot -m bits results/equivalence_classes/gcloud/cpu/*_8-*medium*0.json &
python util/dendrogram.py pyplot -m bits results/equivalence_classes/gcloud/cpu/*_4-*medium*0.json &
python util/dendrogram.py pyplot -m bits results/equivalence_classes/gcloud/cpu/*e-*medium*0.json results/equivalence_classes/gcloud/cpu/*n-*medium*0.json results/equivalence_classes/gcloud/cpu/*ll-*medium-i0.json &
