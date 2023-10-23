import os
import re
import subprocess
import sys

from setuptools import setup

setup(
    name="innfrastructure",
    version="0.2",
    py_modules=["innfrastructure"],
    install_requires=[
        "tensorflow==2.5.0",
        "numpy",
        "prettytable",
        "pandas",
        "PyYAML",
        "py-cpuinfo",
        "click",
        "tabulate",
        "invoke",
        "fabric",
        "psutil",
    ],
    entry_points={
        "console_scripts": [
            "innfcli = innfrastructure.cli:cli",
        ],
    },
    author="Alex Schlögl",
    zip_safe=False,
    extras_require={"test": ["pytest"]},
    packages=['innfrastructure']
)
