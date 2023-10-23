# foreNNsic

This is the experiment repository for our paper "Causes and Effects of Unanticipated Numerical Deviations in Neural Network Inference Frameworks".

This is research code. It may or may not work. If you have any issues, please contact [Alex](https://github.com/alxshine)

## Installation

The code requires python version 3.8.
Ideally, you should isolate everything in a virtual environment, and make sure your `pip` is up-to-date `pip install --upgrade pip`.
Pull and install the infrastructure module
```bash
git submodule update --init
pip install iNNfrastructure
```
Install the additional requirements (`pip install -r requirements.txt`)


## Organization

The experiments all reside in the `experiments` subdirectory.
Each experiment has a `run.py` script, which runs the experiment.

If experiments require preparation, there is a `prepare.py` script.
Additionally, there is a global preparation script at `experiments/prepare.py`, which will prepare models etc.

Scripts used to generate figures in the paper reside in `experiments/figure_scripts`.
These will call the python modules with the correct parameters, and include the output pipes used to create the paper.

Feel free to adjust as necessary, and if you have questions, feel free to reach out :)