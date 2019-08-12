import datetime
import json
import os
import sys
import time
import GPUtil
import subprocess
import socket
import atexit
import click
import colorama
from colorama import Style, Fore, Back
from flufl.lock import Lock

colorama.init()


def get_trial_info(trial_dir, load_start_state=False, load_metric=False):
    start_state_file = os.path.join(trial_dir, 'start_state.json')
    has_start_state = os.path.exists(start_state_file)
    if load_start_state:
        try:
            with open(start_state_file, 'r', encoding='utf-8') as f:
                start_state = json.load(f)
        except FileNotFoundError:
            start_state = None
    else:
        start_state = None

    metric_file = os.path.join(trial_dir, 'metric.tsv')
    has_metric_file = os.path.exists(metric_file)
    if load_metric:
        try:
            metrics = {}
            with open(metric_file, 'r', encoding='ascii') as f:
                for l in f:
                    budget, time, metric = l.strip().split('\t')
                    metrics[int(budget)] = {'time': float(time), 'metric': float(metric)}
        except:
            metrics = None
    else:
        metrics = None

    lock_file = os.path.join(trial_dir, 'lock')
    locked = os.path.exists(lock_file)
    proc_file = os.path.join(trial_dir, 'proc')
    try:
        with open(proc_file, 'r', encoding='ascii') as f:
            machine_id, proc_id = f.read().strip().split('\t')
            proc_status = {'machine_id': int(machine_id),
                           'proc_id': int(proc_id)}
    except FileNotFoundError:
        proc_status = None

    return {'trial_id': os.path.split(trial_dir)[1],
            'trial_dir': trial_dir,
            'has_start_state': has_start_state,
            'has_metrics': has_metric_file,
            'start_state': start_state,
            'metrics': metrics,
            'locked': locked,
            'proc_status': proc_status,
            'should_start': has_start_state and not locked}


class Supervisor:
    def __init__(self, experiment_dir, gpu_per_process, poll_interval, exclude_gpus, gpu_max_mem, gpu_max_load):
        self.experiment_dir = os.path.abspath(experiment_dir)
        self.exclude_gpus = exclude_gpus
        self.gpu_per_process = gpu_per_process
        self.poll_interval = poll_interval
        self.gpu_max_mem = gpu_max_mem
        self.gpu_max_load = gpu_max_load
        self.active_procs = []
        self.last_pending_count = 0
        self.last_active_count = 0

    def print_log(self, *args):
        print(Fore.LIGHTBLACK_EX + f'[{datetime.datetime.now()}]' + Style.RESET_ALL, *args, file=sys.stderr)

    def cleanup(self):
        self.check_completed_procs()
        for proc_info in self.active_procs:
            proc_info['proc'].kill()

            self.log_proc_info(proc_info, True)

    def check_completed_procs(self):
        for proc_info in self.active_procs:
            ret_code = proc_info['proc'].poll()
            if ret_code is not None:
                self.print_log((Fore.RED if ret_code else Fore.GREEN) + 'daemon', proc_info['proc'].pid,
                               'completed with ret code', ret_code, 'for',
                               proc_info['trial_id'], Style.RESET_ALL)
                proc_info['ret_code'] = ret_code
                proc_info['lock'].unlock()
                os.unlink(os.path.join(proc_info['trial_dir'], 'start_state.json'))
                self.log_proc_info(proc_info, False)
        self.active_procs[:] = [p for p in self.active_procs if p['ret_code'] is None]

    def log_proc_info(self, proc_info, killed):
        with open(os.path.join(self.experiment_dir, f'supervisor_{socket.getfqdn()}.json'), 'a',
                  encoding='ascii') as logfile:
            logfile.write(json.dumps({
                'supervisor': os.getpid(),
                'daemon': proc_info['proc'].pid,
                'ret_code': proc_info['ret_code'],
                'start_at': proc_info['start_at'],
                'end_at': time.time(),
                'killed': killed,
                'trial_id': proc_info['trial_id'],
                'mode': proc_info['mode']
            }, ensure_ascii=False))
            logfile.write('\n')

    def supervise(self):
        self.check_completed_procs()

        trials = []
        for parent, dir, files in os.walk(self.experiment_dir):
            trials = [os.path.join(parent, p) for p in dir]
            break
        result = [get_trial_info(p, load_start_state=True) for p in trials]
        pending = [r for r in result if r['should_start']]
        if pending:
            available_gpus = self.get_available_gpus(len(pending))
            if (len(pending) != self.last_pending_count
                    or len(self.active_procs) != self.last_active_count
                    or len(available_gpus)):
                if len(available_gpus):
                    self.print_log(
                        Fore.LIGHTBLUE_EX +
                        f'pending: {len(pending)} active: {len(self.active_procs)} starting: {len(available_gpus)}' +
                        Style.RESET_ALL)
                else:
                    self.print_log(
                        Fore.LIGHTBLACK_EX +
                        f'pending: {len(pending)} active: {len(self.active_procs)}' + Style.RESET_ALL)

            self.last_pending_count = len(pending)
            self.last_active_count = len(self.active_procs)
            for gpu_idx, run_info in zip(available_gpus, pending):
                run_status = self.run_capsule(run_info, gpu_idx)
                if run_status:
                    self.active_procs.append(run_status)
        else:
            if len(pending) != self.last_pending_count or len(self.active_procs) != self.last_active_count:
                self.print_log(
                    Fore.LIGHTBLACK_EX + f'pending: {len(pending)} active: {len(self.active_procs)}' + Style.RESET_ALL)
                self.last_pending_count = len(pending)
                self.last_active_count = len(self.active_procs)

    def run_capsule(self, run_info, gpu_idx):
        lock_path = os.path.join(run_info['trial_dir'], 'lock')
        lock = Lock(lock_path, datetime.timedelta(days=365))
        lock.lock(timeout=datetime.timedelta(seconds=1))
        if not lock.is_locked:
            self.print_log('locking failed for', run_info['trial_id'])
            return None

        # run capsule

        infr_mode = os.environ.get('INFR_MODE', 'train')
        assert infr_mode in ('train', 'debug', 'turbo')

        proc = subprocess.Popen([sys.executable, '-m', run_info['start_state']['module_name']],
                                env={'CUDA_VISIBLE_DEVICES': str(gpu_idx),
                                     'INFR_TRIAL': run_info['trial_id'],
                                     'INFR_EXP_PATH': self.experiment_dir,
                                     'INFR_MODE': infr_mode,
                                     'INFR_REDIRECT_IO': '1',
                                     'INFR_START_STATE': os.path.join(run_info['trial_dir'], 'start_state.json')})

        self.print_log('started daemon', proc.pid, 'for', run_info['trial_id'])

        return {'trial_dir': run_info['trial_dir'],
                'trial_id': run_info['trial_id'],
                'start_at': time.time(),
                'mode': infr_mode,
                'lock': lock,
                'gpu_idx': gpu_idx,
                'proc': proc,
                'pid': proc.pid,
                'ret_code': None}

    def get_available_gpus(self, limit):
        exclude = self.exclude_gpus + [p['gpu_idx'] for p in self.active_procs]
        return GPUtil.getAvailable(order='random',
                                   limit=limit,
                                   maxLoad=self.gpu_max_load,
                                   maxMemory=self.gpu_max_mem,
                                   excludeID=exclude)

    def start(self):
        atexit.register(self.cleanup)
        while True:
            try:
                self.supervise()
                time.sleep(1)
            except KeyboardInterrupt:
                self.print_log('Received stop signal, exiting')
                break


@click.command()
@click.option('--poll-interval', type=int, default=10)
@click.option('--exclude-gpus', default='')
@click.option('--gpu-per-process', type=int, default=1)
@click.option('--gpu-max-mem', type=float, default=0.05)
@click.option('--gpu-max-load', type=float, default=0.1)
@click.argument('folder', required=True)
def run(folder, exclude_gpus, gpu_per_process, poll_interval, gpu_max_mem, gpu_max_load):
    supervisor = Supervisor(folder,
                            gpu_per_process=gpu_per_process,
                            poll_interval=poll_interval,
                            exclude_gpus=[int(idx) for idx in exclude_gpus.split(',')] if exclude_gpus else [],
                            gpu_max_mem=gpu_max_mem,
                            gpu_max_load=gpu_max_load)
    supervisor.start()


if __name__ == '__main__':
    run()
