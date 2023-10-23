from glob import glob
from posixpath import join, basename, splitext

import pandas as pd
import pytest

import common
from innfrastructure import compare, metadata


@pytest.fixture(scope="class")
def df():
    glob_expression = join("results", "equivalence_classes", "*", "cpu", "*.json")
    paths = glob(glob_expression, recursive=True)
    paths = list(filter(lambda p: 'cifar10_medium' in p, paths))
    results = list(map(common.load_result, paths))
    results = list(filter(lambda r: r['cpu_info']['count'] <= 2, results))

    predictions = []
    cpu_stats = []

    rows = []
    for path, result in zip(paths, results):
        cpu_stat = metadata.extract_cpu_stats(result)
        cpu_stat_eqc = compare.get_equivalence_class(cpu_stats, cpu_stat)

        prediction = result['prediction']['bytes']
        eqc = compare.get_equivalence_class(predictions, prediction)

        sample_index, model_type = common.get_model_information(path)

        rows.append({'cpu_stat_eqc': cpu_stat_eqc, 'eqc': eqc, 'model_type': model_type, 'sample_index': sample_index})

    return pd.DataFrame(rows)


def test_single_cpu_stat_for_eqc(df):
    num_unique = df.groupby('eqc').nunique()
    assert not any(num_unique['cpu_stat_eqc'] > 1)


def test_single_eqc_for_cpu_stat(df):
    num_unique = df.groupby(['model_type', 'sample_index', 'cpu_stat_eqc']).nunique()
    assert not any(num_unique['eqc'] > 1)
