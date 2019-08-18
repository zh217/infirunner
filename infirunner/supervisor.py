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
import pprint
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
    def __init__(self, experiment_dir, gpu_per_process, cuda_sync,
                 poll_interval, exclude_gpus, gpu_max_mem, gpu_max_load, mode):
        self.experiment_dir = os.path.abspath(experiment_dir)
        self.exclude_gpus = exclude_gpus
        self.gpu_per_process = gpu_per_process
        self.poll_interval = poll_interval
        self.gpu_max_mem = gpu_max_mem
        self.gpu_max_load = gpu_max_load
        self.active_procs = []
        self.last_pending_count = 0
        self.last_active_count = 0
        self.last_slots = 0
        self.mode = mode
        self.cuda_sync = cuda_sync
        self.slots_file_path = os.path.join(experiment_dir, f'slots_{socket.getfqdn()}')
        os.makedirs(experiment_dir, exist_ok=True)
        if os.path.exists(self.slots_file_path):
            raise RuntimeError('Conflicting slots file at', self.slots_file_path)
        self.slots_file = open(self.slots_file_path, 'w', encoding='ascii')

    def print_log(self, *args):
        print(Fore.LIGHTBLACK_EX + f'[{datetime.datetime.now()}]' + Style.RESET_ALL, *args, file=sys.stderr)

    def update_available_slots(self, n):
        slots_file = self.slots_file
        slots_file.seek(0)
        slots_file.write(f'{n}')
        slots_file.truncate()
        slots_file.flush()

    def cleanup(self):
        self.check_completed_procs()
        try:
            self.slots_file.close()
        except:
            pass
        try:
            os.unlink(self.slots_file_path)
        except:
            pass
        for proc_info in self.active_procs:
            proc_info['proc'].kill()

            self.log_proc_info(proc_info, True)

    def check_completed_procs(self):
        for proc_info in self.active_procs:
            ret_code = proc_info['proc'].poll()
            if ret_code is not None:
                self.print_log((Fore.RED if ret_code else Fore.GREEN) + 'worker', proc_info['proc'].pid,
                               'completed with ret code', ret_code, 'for',
                               proc_info['trial_id'], self.mode, Style.RESET_ALL)
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
                'worker': proc_info['proc'].pid,
                'ret_code': proc_info['ret_code'],
                'start_at': proc_info['start_at'],
                'end_at': time.time(),
                'killed': killed,
                'trial_id': proc_info['trial_id'],
                'mode': self.mode
            }, ensure_ascii=False))
            logfile.write('\n')

    def supervise(self):
        self.check_completed_procs()

        available_gpus = self.get_available_gpus(sys.maxsize)
        trials = []
        for parent, dir, files in os.walk(self.experiment_dir):
            trials = [os.path.join(parent, p) for p in dir]
            break
        result = [get_trial_info(p, load_start_state=True) for p in trials]
        pending = [r for r in result if r['should_start']]
        if pending:
            if (len(pending) != self.last_pending_count
                    or len(self.active_procs) != self.last_active_count
                    or len(available_gpus)
                    or self.last_slots != len(available_gpus)):
                if len(available_gpus):
                    self.print_log(
                        Fore.LIGHTBLUE_EX +
                        f'slots: {len(available_gpus)} pending: {len(pending)} active: {len(self.active_procs)} '
                        f'starting: {min(len(available_gpus), len(pending))}' +
                        Style.RESET_ALL)
                    self.update_available_slots(len(available_gpus) - len(pending))
                else:
                    self.print_log(
                        Fore.LIGHTBLACK_EX +
                        f'slots: {len(available_gpus)} pending: {len(pending)} active: {len(self.active_procs)}' +
                        Style.RESET_ALL)
                    self.update_available_slots(len(available_gpus))

            self.last_pending_count = len(pending)
            self.last_active_count = len(self.active_procs)
            self.last_slots = len(available_gpus)
            for gpu_idx, run_info in zip(available_gpus, pending):
                run_status = self.run_capsule(run_info, gpu_idx)
                if run_status:
                    self.active_procs.append(run_status)
        else:
            if (len(pending) != self.last_pending_count or
                    len(self.active_procs) != self.last_active_count
                    or self.last_slots != len(available_gpus)):
                self.print_log(
                    Fore.LIGHTBLACK_EX +
                    f'slots: {len(available_gpus)} pending: {len(pending)} active: {len(self.active_procs)}' +
                    Style.RESET_ALL)
                self.last_slots = len(available_gpus)
                self.last_pending_count = len(pending)
                self.last_active_count = len(self.active_procs)
                self.update_available_slots(len(available_gpus))

    def run_capsule(self, run_info, gpu_idx):
        lock_path = os.path.join(run_info['trial_dir'], 'lock')
        lock = Lock(lock_path, datetime.timedelta(days=365))
        lock.lock(timeout=datetime.timedelta(seconds=1))
        if not lock.is_locked:
            self.print_log('locking failed for', run_info['trial_id'])
            return None

        # run capsule

        env = {'CUDA_VISIBLE_DEVICES': str(gpu_idx),
               'INFR_TRIAL': run_info['trial_id'],
               'INFR_EXP_PATH': self.experiment_dir,
               'INFR_MODE': self.mode,
               'INFR_REDIRECT_IO': '1',
               'INFR_START_STATE': os.path.join(run_info['trial_dir'], 'start_state.json')}
        if self.cuda_sync:
            env['CUDA_LAUNCH_BLOCKING'] = '1'

        proc = subprocess.Popen([sys.executable, '-m', run_info['start_state']['module_name']],
                                env=env)

        self.print_log('started worker', proc.pid, 'for', run_info['trial_id'], self.mode)

        return {'trial_dir': run_info['trial_dir'],
                'trial_id': run_info['trial_id'],
                'start_at': time.time(),
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

    def check_kill_flags(self):
        for p in self.active_procs:
            if os.path.exists(os.path.join(p['trial_dir'], 'kill')):
                self.print_log(Fore.RED + 'requested SIGKILL', os.path.join(p['trial_dir']))
                p['proc'].kill()
                os.unlink(os.path.join(p['trial_dir'], 'kill'))
            if os.path.exists(os.path.join(p['trial_dir'], 'terminate')):
                self.print_log(Fore.RED + 'requesting SIGTERM', os.path.join(p['trial_dir']))
                p['proc'].terminate()
                os.unlink(os.path.join(p['trial_dir'], 'terminate'))

    def start(self):
        atexit.register(self.cleanup)
        while True:
            try:
                self.supervise()
                self.check_kill_flags()
                time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                self.print_log('Received stop signal, exiting')
                break


@click.command()
@click.option('--poll-interval', type=int, default=5)
@click.option('--exclude-gpus', default='')
@click.option('--gpu-per-process', type=int, default=1)
@click.option('--gpu-max-mem', type=float, default=0.05)
@click.option('--gpu-max-load', type=float, default=0.1)
@click.option('--mode', type=click.Choice(['train', 'debug', 'turbo']), default='train')
@click.option('--cuda-sync/--no-cuda-sync', default=False)
@click.argument('folder', required=True)
def run(folder, exclude_gpus, gpu_per_process, poll_interval, gpu_max_mem, gpu_max_load, mode, cuda_sync):
    pprint.pprint(locals(), stream=sys.stderr)
    supervisor = Supervisor(folder,
                            cuda_sync=cuda_sync,
                            gpu_per_process=gpu_per_process,
                            poll_interval=poll_interval,
                            exclude_gpus=[int(idx) for idx in exclude_gpus.split(',')] if exclude_gpus else [],
                            gpu_max_mem=gpu_max_mem,
                            gpu_max_load=gpu_max_load,
                            mode=mode)
    supervisor.start()


if __name__ == '__main__':
    run()
