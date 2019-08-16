import os
import click
import shutil
import colorama
from colorama import Style, Fore

colorama.init()



@click.command()
@click.option('--dry/--no-dry', default=True)
@click.argument('folder', required=True)
def run(dry, folder):
    folder = os.path.abspath(folder)
    to_remove = []
    for p, dirs, files in os.walk(folder):
        for dir in dirs:
            if not os.path.exists(os.path.join(p, dir, 'metric.tsv')):
                print(Fore.LIGHTRED_EX + 'removing', dir)
                to_remove.append(os.path.join(p, dir))
            # else:
            #     print(Fore.LIGHTBLACK_EX + 'keeping ', dir)
        break

    if not dry:
        for tr in to_remove:
            shutil.rmtree(tr)


if __name__ == '__main__':
    run()
