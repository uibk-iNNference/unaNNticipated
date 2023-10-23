import os
from posixpath import join
from glob import glob
from common import load_result
from innfrastructure import compare


def assert_same_num_eqcs(directories):
    for directory in directories:
        num_eqcs = []
        for distribution_dir, children, _ in os.walk(directory):
            for child in children:
                paths = glob(join(distribution_dir, child, '*.json'))
                results = map(load_result, paths)

                predictions = []
                eqcs = [compare.get_equivalence_class(predictions, result['prediction']['bytes']) for result in results]
                num_eqcs.append(max(eqcs))

        # for all layers we tested, we want the same number of equivalence classes (within a layer), regardless of the
        # parameter distribution
        assert all([eqc == num_eqcs[0] for eqc in num_eqcs])


def test_parameter_distribution_unimportant():
    directories = glob(join("results", "parameter_distribution", "*"), recursive=True)
    assert_same_num_eqcs(directories)


def test_input_distribution_unimportant():
    directories = glob(join("results", "input_distribution", "*"), recursive=True)
    assert_same_num_eqcs(directories)
