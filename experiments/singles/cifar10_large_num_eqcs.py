from innfrastructure import compare, metadata
import json
from glob import glob

def get_eqcs(sample_index):
    paths = glob(f'results/equivalence_classes/gcloud/cpu/*cifar10_large*-i{sample_index}.json')
    results = [json.load(open(p,'r')) for p in paths]
    uniques = []
    eqcs = [compare.get_equivalence_class(uniques, r['prediction']['bytes']) for r in results]
    return max(eqcs)

print(f"Cifar10_large max eqcs: {max([get_eqcs(i) for i in [0,1,6]])}")
