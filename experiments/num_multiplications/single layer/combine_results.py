import os
from typing import List
import click


def make_path_absolute(path: str):
    if os.path.isabs(path):
        return path

    return os.path.join(os.getcwd(), path)


@click.command()
@click.argument("target_dir", type=click.Path(writable=True))
@click.argument(
    "source_dirs",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True),
    nargs=-1,
)
def main(target_dir: str, source_dirs: List[str]):
    os.makedirs(target_dir)

    for source_dir in source_dirs:
        # get subdirectories
        children = os.listdir(source_dir)
        child_paths = [os.path.join(source_dir, c) for c in children]
        sources = [p for p in child_paths if os.path.isdir(p)]

        # make paths absolute
        full_sources = [make_path_absolute(p) for p in sources]
        targets = [os.path.join(target_dir, os.path.basename(s)) for s in sources]
        full_targets = [make_path_absolute(p) for p in targets]

        # softlink (exit if already linked to somewhere)
        for source, target in zip(full_sources, full_targets):
            if os.path.exists(target):
                print(f"Target link {target} already exists, not recreating")
                continue

            os.symlink(source, target)


if __name__ == "__main__":
    main()
