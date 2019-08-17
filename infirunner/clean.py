import math
import os
import click
import shutil
import colorama
from colorama import Style, Fore

from infirunner.watch import FastTSVTail

colorama.init()



@click.command()
@click.option('--dry/--no-dry', default=True)
@click.option('--keep-nan/--no-keep-nan', default=False)
@click.argument('folder', required=True)
def run(dry, folder, keep_nan):
    folder = os.path.abspath(folder)
    to_remove = []
    for p, dirs, files in os.walk(folder):
        for dir in dirs:
            if not os.path.exists(os.path.join(p, dir, 'metric.tsv')):
                print(Fore.LIGHTRED_EX + 'removing empty', dir)
                to_remove.append(os.path.join(p, dir))
            elif not keep_nan:
                with FastTSVTail(os.path.join(p, dir, 'metric.tsv')) as f:
                    budget, metric_time, metric_res = f.tail()
                    if not math.isfinite(float(metric_res)):
                        print(Fore.LIGHTRED_EX + 'removing nan', dir)
                        to_remove.append(os.path.join(p, dir))

            # else:
            #     print(Fore.LIGHTBLACK_EX + 'keeping', dir)
        break

    if not dry:
        for tr in to_remove:
            shutil.rmtree(tr)


if __name__ == '__main__':
    run()
