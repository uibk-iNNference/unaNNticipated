from posixpath import join
from glob import glob
import json

def get_predictions(path):
    with open(path, 'r') as f:
        result = json.load(f)
        predictions = [p['bytes'] for p in result['all_predictions']]
        return predictions


def test_parallelism_deterministic():
    paths = glob(join('results', 'race_conditions', '*json'))
    for path in paths:
        predictions = get_predictions(path)
        assert all([p == predictions[0] for p in predictions]), f"Got indeterministic result for {path}"

test_parallelism_deterministic()
