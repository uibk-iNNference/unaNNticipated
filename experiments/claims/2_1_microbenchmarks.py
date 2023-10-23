import pytest

import common
from glob import glob
from posixpath import join
import pandas as pd

from innfrastructure import compare, metadata


@pytest.fixture(scope="module")
def df():
    glob_expression = join("results", "winning_algorithm", "*", "*.json")
    paths = glob(glob_expression, recursive=True)
    results = list(map(common.load_result, paths))
    rows = []

    predictions = []
    for path, result in zip(paths, results):
        model_type, sample_index = common.get_model_information(path)
        device_name = metadata.get_clean_device_name(result)

        for run_operations, run_predictions in zip(
                result["convolution_operations"], result["all_predictions"]
        ):
            alg_string = "-".join([metadata.get_winning_algorithm(layer['function_calls']) for layer in run_operations])
            final_prediction = run_predictions[-1]["output"]["bytes"]
            final_eqc = compare.get_equivalence_class(predictions, final_prediction)
            rows.append(
                {
                    "model_type": model_type,
                    "sample_index": sample_index,
                    "device_name": device_name,
                    "alg_string": alg_string,
                    "eqc": final_eqc,
                }
            )
    return pd.DataFrame(rows)


def test_alg_choices_define_eqc(df):
    group_by = df.groupby(['model_type', 'sample_index', 'device_name', 'alg_string'])
    assert not any(group_by.nunique().eqc > 1)


def test_algs_define_eqc_medium(df):
    df = df.query('model_type=="cifar10_medium_instrumental"')
    assert all(df.groupby(['sample_index', 'alg_string']).nunique().eqc == 1)


def test_algs_dont_define_eqc_small(df):
    df = df.query('model_type=="cifar10_small_instrumental"')
    assert any(df.groupby(['sample_index', 'alg_string']).nunique().eqc > 1)
