import click
from invoke import run
import os
from os.path import join

import invoke


@click.command()
@click.argument("base_dir", type=str)
def main(base_dir: str):
    # get subdirs
    generator = os.walk(base_dir)
    next(generator)  # skip parent dir
    for (child_dir, _, _) in generator:
        command = f"python {join('main','eval','dendrogram.py')} --base-dir {child_dir} '' -m bits"
        invoke.run(command, disown=True)
    # invoke dendrogram and disown


if __name__ == "__main__":
    main()
