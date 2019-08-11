import click


@click.command()
@click.option('--inactive/--no-inactive', default=False)
@click.option('--poll-interval', type=float, default=2.)
@click.option('--folder')
@click.argument('fields', nargs=-1)
def run(inactive, poll_interval, folder, fields):
    pass
