from innfrastructure.metadata import clean_gpu_name
from glob import glob
import json
from os.path import join, basename
from typing import List

model_types = [
    "minimal",
]
for size in ["small", "medium", "large"]:
    model_types.append(f"cifar10_{size}")



def are_placements_equal(a: List[str], b: List[str]):
    for la, lb in zip(a, b):
        if la != lb:
            return False

    return True


if __name__ == "__main__":
    paths = glob(join("results", "placement", "*.json"))
    results = {}
    for path in paths:
        with open(path, "r") as f:
            setup = basename(path).split("-i_")[0]
            results[setup] = json.load(f)

    for model_type in model_types:
        filtered_results = {
            s: r for s, r in results.items() if r["model_type"] == model_type
        }

        for setup, result in filtered_results.items():
            for other_setup, other_result in filtered_results.items():
                if setup == other_setup:
                    continue

                predictions_equal = (
                    result["prediction"]["bytes"] == other_result["prediction"]["bytes"]
                )
                predictions_text = "equal" if predictions_equal else "UNEQUAL"

                placement_equal = are_placements_equal(
                    result["placements"], other_result["placements"]
                )
                placement_text = "equal" if placement_equal else "UNEQUAL"

                print(
                    f"{setup} and {other_setup} have {predictions_text} predictions with {placement_text} placements"
                )
