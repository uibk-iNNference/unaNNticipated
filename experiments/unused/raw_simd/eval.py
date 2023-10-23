import json
from glob import glob
from posixpath import join
from typing import Dict

import pandas as pd

from innfrastructure import compare

RESULT_DIR = join("results", 'raw_simd')


def main():
    glob_expression = join(RESULT_DIR, '*', '*.json')
    paths = glob(glob_expression, recursive=True)
    results = [load_result(path) for path in paths]

    # get equivalence classes
    unique_predictions = []
    rows = [{'hostname': result['hostname'],
             'equivalence_class': compare.get_equivalence_class(unique_predictions, result['prediction'])}
            for result in
            results]
    assert len(unique_predictions) == 1  # only one equivalence class
    # put into dataframe
    df = pd.DataFrame(rows)


def load_result(path: str) -> Dict:
    with open(path, 'r') as input_file:
        return json.load(input_file)


if __name__ == '__main__':
    main()
