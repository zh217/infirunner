import math
import os
import sys

import click
import shutil
import colorama
from colorama import Style, Fore

from infirunner.watch import FastTSVTail

colorama.init()


@click.command()
@click.option('--dry/--no-dry', default=True)
@click.option('--keep', type=int, default=1)
@click.argument('folder', required=True)
def run(dry, folder, keep):
    folder = os.path.abspath(folder)
    to_remove = []
    for p, dirs, files in os.walk(folder):
        for dir in dirs:
            try:
                saves = os.listdir(os.path.join(p, dir, 'saves'))
                saves = [s for s in saves if all(c in '0123456789' for c in s) and len(s) == 5]
                saves.sort()
                candidates = saves[:-keep]
                for cand in candidates:
                    found = os.path.join(p, dir, 'saves', cand)
                    print(found, file=sys.stderr)
                    to_remove.append(found)
            except FileNotFoundError:
                continue
        break

    if not dry:
        for tr in to_remove:
            shutil.rmtree(tr)


if __name__ == '__main__':
    run()
