import itertools
import json
import os
from posixpath import join
from typing import Dict

from tensorflow import float32, float64, float16

from innfrastructure import metadata, compare

RESULTS_DIR = join('results', 'precision')
TYPE_NAMES = [t.name for t in [
    float32,
    float64,
    float16
]]


def load_result(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def main():
    for subdir, children, filenames in os.walk(RESULTS_DIR):
        if subdir == RESULTS_DIR:
            continue

        results = [load_result(join(subdir, filename)) for filename in filenames]
        predictions = [metadata.convert_json_to_np(result['prediction']) for result in results]

        unique_predictions = []
        eqcs = [compare.get_equivalence_class(unique_predictions, prediction) for prediction in predictions]
        num_eqcs = max(eqcs) + 1

        remaining_precisions = [compare.remaining_precision(a, b) for a, b in itertools.combinations(predictions, 2)]

        print(f"{subdir}: {num_eqcs} equivalence classes, min remaining precision: {min(remaining_precisions)}")


if __name__ == '__main__':
    main()
