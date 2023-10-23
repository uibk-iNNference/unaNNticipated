# coding: utf-8
from glob import glob
import json
from innfrastructure import metadata, compare
import itertools

paths = glob('results/equivalence_classes/gcloud/cpu/*cifar10_medium-i0.json')
results = [json.load(open(p, 'r')) for p in paths]
predictions = [metadata.convert_json_to_np(r['prediction']) for r in results]
device_names = [f"{metadata.get_clean_device_name(r)}-{r['cpu_info']['family']}-{r['cpu_info']['model']}" for r in results]
combinations = list(itertools.combinations(zip(device_names, predictions), 2))

max_name = None
max_dist = 0

for (name_a, pred_a), (name_b, pred_b) in combinations:
    distance = compare.l2_distance(pred_a, pred_b)
    if distance > max_dist:
        max_name = f"{name_a} -- {name_b}"
        max_dist = distance

print(max_name)
print(max_dist)
