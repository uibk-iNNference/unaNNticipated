import json
from posixpath import basename, splitext


def get_model_information(path: str):
    filename = splitext(basename(path))[0]
    parts = filename.split('-')
    sample_index = parts[-1]
    sample_index = int(sample_index[1:])

    model_type = parts[-2]
    return model_type, sample_index


def load_result(path: str):
    with open(path, 'r') as input_file:
        result = json.load(input_file)
    return result
