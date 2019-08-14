import abc
import enum
import importlib
import math
import os
import sys
import time
import datetime

import colorama
import click
import infirunner.capsule

import infirunner.param as param
from collections import namedtuple
from colorama import Style, Fore, Back

from infirunner.util import make_trial_id, UniformBernoulli
from infirunner.generator import Generator
from infirunner.watch import ExperimentWatcher

colorama.init()


def int_ceil(x):
    return int(math.ceil(x))


def int_floor(x):
    return int(math.floor(x))


class ElementVerdict(enum.Enum):
    GOOD = 'good'
    BAD = 'bad'
    UNKNOWN = 'unknown'


def log_print(*args):
    print(Fore.LIGHTBLACK_EX + f'[{datetime.datetime.now()}]' + Style.RESET_ALL, *args, file=sys.stderr)


# Note that the metric is always assumed to be optimized towards minimum.
class BracketElement:
    def __init__(self, bracket, round, budget, metric, trial, active):
        self.bracket = bracket
        self.round = round
        self.budget = budget
        self.metric = metric
        self.trial = trial
        self.active = active

    def __repr__(self):
        return (f'<BracketElement bracket={self.bracket} round={self.round} budget={self.budget} '
                f'metric={self.metric} trial={self.trial} active={self.active}>')


class ParamGen(abc.ABC):
    def __init__(self, module):
        importlib.import_module(module)
        self.capsule = infirunner.capsule.active_capsule

    def get_next_parameter(self):
        return self.capsule.gen_params(use_default=False, skip_const=True)


class RandomParamGen(ParamGen):
    pass


class BOHBParamGen(ParamGen):
    def __init__(self, module, experiment_dir, random_ratio, sample_size, result_size_threshold=None, good_ratio=0.15,
                 model_cache_time=30, mode='minimize'):
        super().__init__(module)

        if result_size_threshold is None:
            sample_params = super().get_next_parameter()
            result_size_threshold = len(sample_params) + 1

        self.experiment_dir = experiment_dir
        self.dice = UniformBernoulli(random_ratio)
        self.good_ratio = good_ratio
        self.sample_size = sample_size
        self.result_size_threshold = result_size_threshold
        self.model_cache_time = model_cache_time
        self.last_stats_collect_time = 0.
        self.last_stats = (None, None)
        self.last_models = (None, None)
        self.is_maximize = mode == 'maximize'

    def get_suggested_next_parameter(self, goods, bads):
        good_model, bad_model = self.last_models
        if good_model is None or bad_model is None:
            good_model = build_it()
            bad_model = build_it()
            self.last_models = good_model, bad_model

        candidates = [super().get_next_parameter() for _ in range(self.sample_size)]

        # use model to select the best one and return it

    def collect_stats(self):
        now = time.time()
        if now - self.last_stats_collect_time > self.model_cache_time:
            metrics = list(self.get_all_budget_metrics().items())
            metrics.sort(key=lambda x: len(x[1]), reverse=True)
            goods = None
            bads = None

            for budget, trial_data in metrics:
                bads_ = [(trial, metric) for trial, metric in trial_data if
                         metric is None or not math.isfinite(metric)]
                goods_ = [(trial, metric) for trial, metric in trial_data if
                          metric is not None and math.isfinite(metric)]
                goods_.sort(key=lambda x: x[1], reverse=self.is_maximize)
                good_size = int_ceil(len(goods_) * self.good_ratio)
                bads_ = goods_[good_size:] + bads_
                goods_ = goods_[:good_size]
                if len(bads_) >= self.result_size_threshold and len(goods_) >= self.result_size_threshold:
                    log_print(Fore.LIGHTBLACK_EX, f'collected stats for budget {budget} with {len(goods_)} goods, '
                                                  f'{len(bads_)} bads')
                    log_print(Fore.LIGHTBLACK_EX, f'best good: {goods_[0][1]:10.4f}, best bad: {bads_[0][1]:10.4f}')
                    goods = goods_
                    bads = bads_
                    break
            if self.last_stats != (goods, bads):
                self.last_stats = (goods, bads)
                self.last_models = (None, None)
            self.last_stats_collect_time = now
            return goods, bads
        else:
            return self.last_stats

    def get_all_budget_metrics(self):
        active_dirs = []
        for parent, dirs, files in os.walk(self.experiment_dir):
            active_dirs = dirs
            break

        metrics = {}

        for dir in active_dirs:
            try:
                with open(os.path.join(self.experiment_dir, dir, 'metric.tsv'), 'rb') as f:
                    for l in f:
                        budget, metric_time, metric_res = l.split(b'\t')
                        budget_metric = metrics.setdefault(int(budget), [])
                        budget_metric.append((dir, float(metric_res)))
            except FileNotFoundError:
                continue
        return metrics

    def get_next_parameter(self):
        if self.dice():
            return super().get_next_parameter()
        else:
            goods, bads = self.collect_stats()
            if goods and bads:
                return self.get_suggested_next_parameter(goods, bads)


class Hyperband:
    def __init__(self, min_budget, max_budget, reduction_ratio=math.e):
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.reduction_ratio = reduction_ratio
        self.bracket_max = int_floor(math.log(max_budget / min_budget, reduction_ratio))
        self.brackets = self.make_brackets()
        self.cur_bracket_idx = 0
        self.cur_round_idx = 0

    def pprint_brackets(self):
        for bracket_idx, bracket in enumerate(self.brackets):
            log_print(Fore.LIGHTBLACK_EX + f'bracket {bracket_idx}:')
            for round_idx, round in enumerate(bracket):
                actives = [e for e in round if e.active]
                dones = [e for e in round if e.metric is not None]
                budget = round[0].budget
                to_print = (f'\tround {round_idx:1}: {len(round):3} trials with {budget:3} budgets, ' +
                            f'{len(actives):3} active, {len(dones):3} complete')
                if dones:
                    best_metric = min(e.metric for e in dones if math.isfinite(e.metric))
                    to_print += f', {best_metric:10.4f} best'
                else:
                    to_print = Fore.LIGHTBLACK_EX + to_print
                log_print(to_print)

    def make_brackets(self):
        brackets = []
        for bracket_idx in range(self.bracket_max, -1, -1):
            bracket = []
            init_n_trials = (self.bracket_max + 1) / (bracket_idx + 1) * (self.reduction_ratio ** bracket_idx)
            init_budget = self.max_budget / (self.reduction_ratio ** bracket_idx)

            for i in range(bracket_idx + 1):
                n_trials = int_ceil(init_n_trials / (self.reduction_ratio ** i))
                if i == bracket_idx:
                    budget = self.max_budget
                elif bracket_idx == self.bracket_max and i == 0:
                    budget = self.min_budget
                    n_trials = int_ceil(self.max_budget / self.min_budget)
                else:
                    budget = int_ceil(init_budget * (self.reduction_ratio ** i))
                bracket_trials = []
                for _ in range(n_trials):
                    bracket_trials.append(BracketElement(bracket=self.bracket_max - bracket_idx,
                                                         round=i,
                                                         budget=budget,
                                                         metric=None,
                                                         trial=None,
                                                         active=False))
                bracket.append(bracket_trials)
            brackets.append(bracket)
        return brackets

    def is_round_complete(self, bracket_id, round_id):
        return all(e.trial is not None and not e.active for e in self.brackets[bracket_id][round_id])

    def is_complete(self):
        return all(e.trial is not None and not e.active for e in self.brackets[-1][-1])

    def request_trial(self):
        # if all brackets are complete, raise StopIteration
        # if caller should wait, return None
        # return: BracketElement, note that if el.trial is empty the caller is responsible for filling it
        if self.cur_bracket_idx > self.bracket_max:
            raise StopIteration
        cur_bracket = self.brackets[self.cur_bracket_idx]
        cur_round = cur_bracket[self.cur_round_idx]

        inactive_without_trial = [e for e in cur_round if e.trial is None and not e.active]
        if inactive_without_trial:
            ret = inactive_without_trial[0]

            if self.cur_round_idx == 0:
                ret.active = True
                return ret
            else:
                cur_round_trial_ids = set(e.trial for e in cur_round if e.trial is not None)
                last_round_completed_trials = [e for e in cur_bracket[self.cur_round_idx - 1]
                                               if e.metric is not None and math.isfinite(e.metric)
                                               and e.trial not in cur_round_trial_ids]
                if last_round_completed_trials:
                    # this COULD be empty, since all elements in previous round may have failed with NaN
                    # in that case the caller should initialize a new one to run
                    last_round_completed_trials.sort(key=lambda e: e.metric)
                    best_available_trial = last_round_completed_trials[0]
                    log_print(Fore.LIGHTBLACK_EX + 'promote best available trial', best_available_trial.trial,
                              best_available_trial.metric,
                              '(worst is', last_round_completed_trials[-1].metric, ')')
                    ret.trial = best_available_trial.trial

                ret.active = True
                # if no trial is present, the caller is responsible for filling it it
                return ret

        if self.is_round_complete(self.cur_bracket_idx, self.cur_round_idx):
            if self.cur_round_idx == len(cur_bracket) - 1:
                self.cur_round_idx = 0
                self.cur_bracket_idx += 1
            else:
                self.cur_round_idx += 1
            log_print(Fore.LIGHTBLACK_EX + str(id(self)), 'proceed to bracket', self.cur_bracket_idx, 'round',
                      self.cur_round_idx)
            return self.request_trial()
        else:
            return None

    def report_trial(self, bracket_idx, round_idx, trial, metric):
        # mark inactive, set metric
        # make verdict for all completed rounds
        if math.isfinite(metric):
            log_print('hyperband received report', bracket_idx, round_idx, trial, metric)
        else:
            log_print(Fore.LIGHTRED_EX + 'hyperband received report', bracket_idx, round_idx, trial, metric)
        requested_round = self.brackets[bracket_idx][round_idx]
        requested_element = None
        for el in requested_round:
            if el.trial == trial:
                requested_element = el
        assert requested_element
        requested_element.metric = metric
        requested_element.active = False


class HyperbandDriver:
    def __init__(self, experiment_dir, trial_generator, param_generator, min_budget, max_budget,
                 reduction_ratio, sleep_interval, max_hyperbands, mode):
        self.experiment_dir = experiment_dir
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.reduction_ratio = reduction_ratio
        self.sleep_interval = sleep_interval
        self.watcher = ExperimentWatcher(experiment_dir)
        self.hyperbands = []
        self.watch_active_trials = []
        self.trial_generator = trial_generator
        self.param_generator = param_generator
        self.max_hyperbands = max_hyperbands
        self.is_maximize = mode == 'maximize'

    def generate_new_trial(self, end_budget, n_gpu=1):
        params = self.param_generator.get_next_parameter()
        new_id = make_trial_id()
        log_print(Fore.LIGHTGREEN_EX + f'generate new trial {new_id} with budget 0 -> {end_budget}')
        self.trial_generator.change_capsule_trial_id(new_id)
        self.trial_generator.save_start_state(start_budget=0,
                                              end_budget=end_budget,
                                              n_gpu=n_gpu,
                                              params=params)
        return new_id

    def amend_trial(self, old_trial, end_budget, n_gpu=1):
        self.trial_generator.change_capsule_trial_id(old_trial)
        new_state = self.trial_generator.amend_start_state(end_budget=end_budget, n_gpu=n_gpu)
        log_print(
            Fore.LIGHTBLUE_EX + f'amended trial {old_trial} with budget {new_state["start_budget"]} -> {new_state["end_budget"]}')

    def get_next_hyperband_trial(self):
        for hyperband_idx, hyperband in enumerate(self.hyperbands):
            try:
                new_trial = hyperband.request_trial()
            except StopIteration:
                continue
            if new_trial is not None:
                if new_trial.trial is None:
                    new_trial.trial = self.generate_new_trial(new_trial.budget)
                else:
                    self.amend_trial(new_trial.trial, new_trial.budget)
                return hyperband_idx, new_trial
        if len(self.hyperbands) < self.max_hyperbands:
            self.hyperbands.append(Hyperband(min_budget=self.min_budget,
                                             max_budget=self.max_budget,
                                             reduction_ratio=self.reduction_ratio))
            return self.get_next_hyperband_trial()
        return None, None

    def get_available_slots(self):
        slot_files = []
        for parent, dirs, files in os.walk(self.experiment_dir):
            slot_files = [f for f in files if f.startswith('slots_')]
            break
        total_slots = 0
        for slot_file in slot_files:
            with open(os.path.join(self.experiment_dir, slot_file), 'rb') as f:
                total_slots += int(f.read().strip())
        return total_slots

    def check_for_completed_trials(self):
        completed_trials = set()
        watcher_result = {k: v for k, v in
                          self.watcher.poll(slots=False, only=[t.trial for _, t in self.watch_active_trials],
                                            fields=False)['trials'] if not v['active']}
        for hyperband_idx, trial in self.watch_active_trials:
            if trial.trial in watcher_result:
                completed_trials.add(trial.trial)
                trial_result = watcher_result[trial.trial]
                log_print(Fore.LIGHTBLACK_EX + f'obtained watcher result for {trial.trial}')
                if trial_result['budget'] != trial.budget:
                    trial_result['metric'] = float('nan')
                if trial_result['metric'] is None:
                    trial_result['metric'] = float('nan')
                metric = trial_result['metric']
                if self.is_maximize:
                    metric = -metric
                self.hyperbands[hyperband_idx].report_trial(trial.bracket, trial.round, trial.trial, metric)
        self.watch_active_trials[:] = [t for t in self.watch_active_trials if t[1].trial not in completed_trials]

    def start_trials(self):
        n_slots = self.get_available_slots()
        for _ in range(n_slots):
            hyperband_idx, new_trial = self.get_next_hyperband_trial()
            if new_trial is not None:
                # launch new trial
                # add to trials being watched
                log_print(Fore.LIGHTBLACK_EX + f'watching trial {new_trial.trial} of band {hyperband_idx}')
                self.watch_active_trials.append((hyperband_idx, new_trial))

    def start(self):
        last_watching = set()
        while True:
            self.check_for_completed_trials()
            self.start_trials()
            cur_watching = set(t.trial for _, t in self.watch_active_trials)
            if last_watching != cur_watching:
                for idx, hb in enumerate(self.hyperbands):
                    log_print(Fore.LIGHTBLACK_EX + '----- Hyperband', idx, id(hb), '-----')
                    hb.pprint_brackets()
            last_watching = cur_watching
            if len(self.hyperbands) == self.max_hyperbands and all(hb.is_complete() for hb in self.hyperbands):
                break
            time.sleep(self.sleep_interval)


@click.command()
@click.option('--exp-path', default='_exp')
@click.option('--module', required=True)
@click.option('--min-budget', type=int, default=1)
@click.option('--max-budget', type=int, required=True)
@click.option('--sleep-interval', type=float, default=10.)
@click.option('--max-hyperbands', type=int, default=2)
@click.option('--reduction-ratio', type=float, default=math.e)
@click.option('--mode', type=click.Choice(['maximize', 'minimize']), default='minimize')
def run(module, exp_path, min_budget, max_budget, reduction_ratio, max_hyperbands, sleep_interval, mode):
    exp_path = os.path.abspath(exp_path)
    trial_gen = Generator(module, exp_path)
    param_gen = ParamGen(module)
    driver = HyperbandDriver(experiment_dir=exp_path,
                             trial_generator=trial_gen,
                             param_generator=param_gen,
                             min_budget=min_budget,
                             max_budget=max_budget,
                             reduction_ratio=reduction_ratio,
                             sleep_interval=sleep_interval,
                             max_hyperbands=max_hyperbands,
                             mode=mode)
    driver.start()


if __name__ == '__main__':
    run()
