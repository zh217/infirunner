import click


class Supervisor:
    def __init__(self, experiment_dir, n_gpu, gpu_per_process, poll_interval):
        self.experiment_dir = experiment_dir
        self.n_gpu = n_gpu
        self.gpu_per_process = gpu_per_process
        self.poll_interval = poll_interval

    def start(self):
        pass


@click.command()
@click.option('--poll-interval', type=int, default=10)
@click.option('--n-gpu', type=int)
@click.option('--gpu-per-process', type=int, default=1)
@click.argument('folder', required=True)
def run(folder, n_gpu, gpu_per_process, poll_interval):
    supervisor = Supervisor(folder)
    supervisor.start()


if __name__ == '__main__':
    run()
