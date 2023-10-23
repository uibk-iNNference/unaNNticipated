# iNNfrastructure

This is the infrastructure of our NeurIPS paper, and handles the cloud orchestration as well as the inference used for our experiments.
This is research code, and may or may not work.
If you have any questions, please contact [Alex](https://github.com/alxshine).

## Installation

Ideally, create a virtual environment for isolation.
Also make sure that you update your pip to avoid dependency conflicts: `pip install --upgrade pip`

Afterwards, install the package using `pip install .`

## Usage

### CLI

We provide the `innfcli` command for quick and standardized prediction.
For usage, please see the [foreNNsic experiment repository](TODO) and the run files there.

The CLI only includes code to interact with GCP. If you want to use AWS, you need to use the API directly. Our experiments mostly use the `ensure_configs_running` function from both GCP and AWS modules.

### API

The API is provided by the `innfrastructure` package.
The functions are documented in the source code.
