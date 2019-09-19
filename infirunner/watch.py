import json
import math
import os
import sys
import time
import colorama
from colorama import Style, Fore, Back

colorama.init()

import click


def clear_screen():
    print('\x1b[2J', file=sys.stderr)


class FastTSVTail:
    def __init__(self, filepath, seek_length=128):
        self.filepath = filepath
        self.seek_length = seek_length
        self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._file:
            try:
                self._file.close()
            except:
                pass

    def __del__(self):
        self.close()

    @property
    def file(self):
        if not self._file:
            self._file = open(self.filepath, 'rb')
        return self._file

    def tail(self):
        try:
            self.file.seek(-self.seek_length, os.SEEK_END)
        except OSError:
            self.file.seek(0)

        lines = self.file.readlines()
        for l in reversed(lines):
            l_stripped = l.strip()
            if l_stripped:
                return l_stripped.split(b'\t')
        return []


class TrialWatcher:
    def __init__(self, trial_dir, watch_fields):
        self.trial_dir = trial_dir
        self.watch_fields = watch_fields

    def close(self):
        pass

    def poll_field(self, field):
        try:
            with FastTSVTail(os.path.join(self.trial_dir, 'logs', field + '.tsv')) as f:
                steps, time, relative, value = f.tail()
                steps = int(steps)
                time = float(time)
                relative = float(relative)
                value = float(value)
                return {'steps': steps,
                        'time': time,
                        'relative': relative,
                        'value': value}
        except FileNotFoundError:
            pass
        except Exception as e:
            print(self.trial_dir, e, type(e))

    def poll(self, metric=True, fields=True):
        budget = None
        metric_time = None
        metric_res = None

        if metric:
            try:
                with FastTSVTail(os.path.join(self.trial_dir, 'metric.tsv')) as f:
                    budget, metric_time, metric_res = f.tail()
                    budget = int(budget)
                    metric_time = float(metric_time)
                    metric_res = float(metric_res)
            except FileNotFoundError:
                pass
            except Exception as e:
                print(self.trial_dir, e, type(e))

        if fields:
            fields_res = {field: self.poll_field(field) for field in self.watch_fields}
        else:
            fields_res = None

        return {
            'active': os.path.exists(os.path.join(self.trial_dir, 'lock')),
            'time': metric_time,
            'budget': budget,
            'metric': metric_res,
            'fields': fields_res
        }


class ExperimentWatcher:
    def __init__(self, experiment_dir, watch_fields=()):
        self.experiment_dir = experiment_dir
        self.watch_fields = watch_fields
        self.watchers = {}

    def poll(self, slots=True, only=None, **kwargs):
        active_dirs = set()
        slot_files = []
        for parent, dirs, files in os.walk(self.experiment_dir):
            active_dirs = set(dirs)
            slot_files = [file for file in files if file.startswith('slots_')]
            break
        watching_dirs = set(self.watchers.keys())
        unwatched = active_dirs - watching_dirs
        deleted = watching_dirs - active_dirs
        for trial in deleted:
            self.watchers[trial].close()
            del self.watchers[trial]
        for trial in unwatched:
            self.watchers[trial] = TrialWatcher(os.path.join(self.experiment_dir, trial), self.watch_fields)
        if slots:
            supervisor_slots = {}
            total_slots = 0
            for slot_file in slot_files:
                try:
                    with open(os.path.join(self.experiment_dir, slot_file), 'rb') as f:
                        n_slot = int(f.read().strip())
                        total_slots += n_slot
                        supervisor_slots[slot_file[6:]] = n_slot
                except Exception as e:
                    print(e, file=sys.stderr)
        else:
            total_slots = None
            supervisor_slots = None

        ret_trials_keys = sorted(self.watchers.keys())
        if only is not None:
            only = set(only)
            ret_trials_keys = [k for k in ret_trials_keys if k in only]
        return {
            'total_slots': total_slots,
            'slots': supervisor_slots,
            'trials': [(k, self.watchers[k].poll(**kwargs)) for k in ret_trials_keys]
        }

    def get_trial_last_state(self, trial):
        with open(os.path.join(self.experiment_dir, trial, 'last_state.json'), 'r', encoding='utf-8') as f:
            return json.load(f)


def pprint_result(result, fields, limit, threshold, hide_inactive, ignore_inactive_nan):
    slots_line = f'TOTAL SLOTS: {Fore.LIGHTBLUE_EX}{result["total_slots"]}' + ' ' * 2
    for k, v in result['slots'].items():
        slots_line += f'{Style.RESET_ALL}{k}: {Fore.LIGHTBLUE_EX}{v} '
    print(slots_line + Style.RESET_ALL)
    print()
    title = f'{"TRIAL":24} {"BUDGET":>6} {"METRIC":>12} {"STEP":>8} {"REL":>7}'
    for field in fields:
        title += f' {field[:12]:>12}'
    print(title)
    print(Fore.LIGHTBLACK_EX + ('-' * len(title)))
    trials_data = result['trials']
    if hide_inactive:
        trials_data = [t for t in trials_data if t[1]['active']]
    elif ignore_inactive_nan:
        trials_data = [t for t in trials_data if
                       t[1]['active'] or (t[1]['metric'] is not None and math.isfinite(t[1]['metric']))]
    if threshold is not None:
        trials_data = [t for t in trials_data if
                       t[1]['active'] or (t[1]['metric'] is not None and t[1]['metric'] < threshold)]
        trials_data.sort(key=lambda d: (d[1]['active'], -d[1]['metric'], d[1]['budget'] or -1, d[0]))
    else:
        trials_data.sort(key=lambda d: (d[1]['active'], d[1]['budget'] or -1, d[0]))
    if limit:
        trials_data = trials_data[-limit:]
    for trial, data in trials_data:
        line = f'{trial:24}'
        if data['active']:
            line = Fore.LIGHTGREEN_EX + line + Style.RESET_ALL
        else:
            line = Fore.LIGHTBLACK_EX + line + Style.RESET_ALL
        try:
            metric = data['metric']
            budget = data['budget']
            if budget is None:
                budget = 0
            if metric is None:
                metric = float('nan')
        except:
            budget = 0
            metric = float('nan')
        metric_str = f' {metric:12.5f}'
        if len(metric_str) != 13:
            metric_str = f' {metric:>12.5e}'
        line += f' {budget:6}{metric_str} '
        subline = ''
        max_steps = 0
        relative = 0
        for field in fields:
            try:
                max_steps = max(max_steps, data['fields'][field]['steps'])
                value = data["fields"][field]["value"]
                relative = max(relative, data["fields"][field]["relative"])
            except:
                value = float('nan')
            val_str = f' {value:12.5f}'
            if len(val_str) != 13:
                val_str = f' {value:>12.5e}'
            subline += val_str
        print(line + f'{max_steps:>8} {int(relative // 3600):>4}:{int((relative // 60) % 60):02}' + subline)
    print(Fore.LIGHTBLACK_EX + ('-' * len(title)))
    print(title)


@click.command()
@click.option('--fields', default='')
@click.option('--hide-inactive/--no-hide-inactive', default=False)
@click.option('--only')
@click.option('--watch', type=float, default=0.)
@click.option('--limit', type=int)
@click.option('--threshold', type=float)
@click.option('--ignore-inactive-nan/--no-ignore-inactive-nan', default=True)
@click.argument('folder', required=True)
def run(folder, fields, only, watch, limit, hide_inactive, ignore_inactive_nan, threshold):
    fields = [field for field in fields.split(',') if field]
    if only:
        only = [t for t in only.split(',') if t]

    watcher = ExperimentWatcher(folder, fields)
    if watch:
        while True:
            clear_screen()
            pprint_result(watcher.poll(only=only), fields, limit, threshold, hide_inactive, ignore_inactive_nan)
            time.sleep(watch)
    else:
        pprint_result(watcher.poll(only=only), fields, limit, threshold, hide_inactive, ignore_inactive_nan)


if __name__ == '__main__':
    run()
