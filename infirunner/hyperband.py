import abc
import enum
import importlib
import json
import math
import os
import random
import sys
import time
import datetime

import colorama
import click
import infirunner.capsule
import numpy as np
import scipy.stats as sps

import infirunner.param as param
from collections import namedtuple
from colorama import Style, Fore, Back
from statsmodels.nonparametric.api import KDEMultivariate

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
    def __init__(self, module, experiment_dir,
                 random_ratio, random_sample_size,
                 guided_ratio, guided_sample_size,
                 result_size_threshold=None, good_ratio=0.15,
                 model_cache_time=30, mode='minimize',
                 min_bandwidth=1e-3, bandwidth_estimation='normal_reference',
                 bandwidth_factor=3.):
        super().__init__(module)

        assert 0 <= random_ratio <= 1
        assert 0 <= guided_ratio <= 1
        assert 0 <= random_ratio + guided_ratio <= 1

        sample_params = super().get_next_parameter()
        if result_size_threshold is None:
            result_size_threshold = len(sample_params) + 1
        else:
            result_size_threshold = max(len(sample_params) + 1, result_size_threshold)
        log_print(Fore.LIGHTBLACK_EX + 'model-based threshold is', result_size_threshold)

        self.experiment_dir = experiment_dir
        self.random_dice = UniformBernoulli(random_ratio)
        self.guided_dice = UniformBernoulli(guided_ratio / (1 - random_ratio))
        self.good_ratio = good_ratio
        self.random_sample_size = random_sample_size
        self.guided_sample_size = guided_sample_size
        self.result_size_threshold = result_size_threshold
        self.model_cache_time = model_cache_time
        self.last_stats_collect_time = 0.
        self.last_stats = (None, None)
        self.last_models = (None, None)
        self.is_maximize = mode == 'maximize'
        self.min_bandwidth = min_bandwidth
        self.bandwidth_estimation = bandwidth_estimation
        self.bandwidth_factor = bandwidth_factor
        self.kde_vartypes, self.kde_data_encoder, self.kde_data_decoder, self.kde_data_bounds = self.make_kde_helpers()

    def make_kde_helpers(self):
        param_keys = list(super().get_next_parameter().keys())
        param_keys.sort()
        param_gen = self.capsule.param_gen

        var_types = ''.join(param_gen[key].var_type for key in param_keys)

        data_bounds = [param_gen[key].encoded_bounds() for key in param_keys]

        def data_encoder(data):
            return [param_gen[key].encode_as_numerical(data[key]) for key in param_keys]

        def data_decoder(data, old_params):
            ret = {}
            for idx, key in enumerate(param_keys):
                decoded = param_gen[key].decode_from_numerical(data[idx])
                if var_types[idx] == 'u' and decoded != old_params[key]:
                    while True:
                        decoded = param_gen[key].get_next_value()
                        if decoded != old_params[key]:
                            break
                ret[key] = decoded
            return ret

        return var_types, data_encoder, data_decoder, data_bounds

    def get_trial_params(self, trial):
        with open(os.path.join(self.experiment_dir, trial, f'last_state.json'), 'r', encoding='utf-8') as f:
            old_params = json.load(f)['params']
        return old_params

    def guided_modify_parameter(self, trial, model):
        old_params = self.get_trial_params(trial)
        old_params_encoded = self.kde_data_encoder(old_params)
        new_params_encoded = []
        for old_param, bw_, (lbound, ubound), vartype in zip(old_params_encoded, model.bw, self.kde_data_bounds,
                                                             self.kde_vartypes):
            bw = bw_ * self.bandwidth_factor
            if lbound is None or ubound is None:
                new_params = np.random.normal(loc=old_param, scale=bw)
            else:
                new_params = sps.truncnorm.rvs((lbound - old_param) / bw, (ubound - old_param) / bw,
                                               loc=old_param, scale=bw)
            new_params_encoded.append(new_params)

        return self.kde_data_decoder(new_params_encoded, old_params)

    def get_suggested_next_parameter(self, goods, bads):
        good_model, bad_model = self.last_models
        if good_model is None or bad_model is None:
            good_model = KDEMultivariate(data=[self.kde_data_encoder(self.get_trial_params(t)) for t, _ in goods],
                                         var_type=self.kde_vartypes,
                                         bw=self.bandwidth_estimation)
            bad_model = KDEMultivariate(data=[self.kde_data_encoder(self.get_trial_params(t)) for t, _ in bads],
                                        var_type=self.kde_vartypes,
                                        bw=self.bandwidth_estimation)
            good_model.bw = np.clip(good_model.bw, self.min_bandwidth, None)
            bad_model.bw = np.clip(bad_model.bw, self.min_bandwidth, None)
            self.last_models = good_model, bad_model

        best_score = float('-inf')
        best_candidate = None
        use_guided = self.guided_dice()
        for _ in range(self.guided_sample_size if use_guided else self.random_sample_size):
            if use_guided:
                next_param = self.guided_modify_parameter(random.choice(goods)[0], good_model)
            else:
                next_param = super().get_next_parameter()
            good_score = np.log(np.clip(good_model.pdf(self.kde_data_encoder(next_param)), 1e-32, None))
            bad_score = np.log(np.clip(bad_model.pdf(self.kde_data_encoder(next_param)), 1e-32, None))
            score = good_score - bad_score
            if score > best_score:
                best_score = score
                best_candidate = next_param
        log_print(Fore.LIGHTBLACK_EX + 'proposing', 'guided' if use_guided else 'sieved', 'parameter with score',
                  best_score)
        return best_candidate

    def collect_stats(self):
        now = time.time()
        if now - self.last_stats_collect_time > self.model_cache_time:
            metrics = list(self.get_all_budget_metrics().items())
            metrics.sort(key=lambda x: x[0], reverse=True)
            goods = None
            bads = None

            for budget, trial_data in metrics:
                bads_ = [(trial, metric) for trial, metric in trial_data if
                         metric is None or not math.isfinite(metric)]
                goods_ = [(trial, metric) for trial, metric in trial_data if
                          metric is not None and math.isfinite(metric)]
                if len(goods_) >= self.result_size_threshold and len(goods_) + len(bads_) > self.result_size_threshold:
                    goods_.sort(key=lambda x: x[1], reverse=self.is_maximize)
                    good_size = int_ceil(len(goods_) * self.good_ratio)
                    bads_ = list(reversed(goods_))[
                            :max(len(goods_) - good_size, self.result_size_threshold - len(bads_))] + bads_
                    goods_ = goods_[:max(good_size, self.result_size_threshold)]
                    log_print(Fore.LIGHTBLACK_EX + f'collected stats for budget {budget} with {len(goods_)} goods, '
                                                   f'{len(bads_)} bads')
                    log_print(Fore.LIGHTBLACK_EX + f'best good: {goods_[0][1]:10.4f}, best bad: {bads_[0][1]:10.4f}')
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

        nan_metrics = []

        for dir in active_dirs:
            try:
                with open(os.path.join(self.experiment_dir, dir, 'metric.tsv'), 'rb') as f:
                    for l in f:
                        budget, metric_time, metric_res = l.split(b'\t')
                        budget_metric = metrics.setdefault(int(budget), [])
                        metric = float(metric_res)
                        budget_metric.append((dir, metric))
                        if not math.isfinite(metric):
                            nan_metrics.append((budget, dir))
            except FileNotFoundError:
                continue
        for budget in metrics.keys():
            for trial_budget, dir in nan_metrics:
                if budget > trial_budget:
                    metrics[budget].append((dir, float('nan')))
        return metrics

    def get_next_parameter(self):
        if self.random_dice():
            log_print(Fore.LIGHTBLACK_EX + 'generate random parameter because dice says so')
            return super().get_next_parameter()
        else:
            goods, bads = self.collect_stats()
            if goods and bads:
                return self.get_suggested_next_parameter(goods, bads)
            else:
                log_print(Fore.LIGHTBLACK_EX + 'generate random parameter because not enough samples')
                return super().get_next_parameter()


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
                good_dones = [e for e in dones if math.isfinite(e.metric)]
                budget = round[0].budget
                to_print = (f'\tround {round_idx:1}: {len(round):3} trials with {budget:3} budgets, ' +
                            f'{len(actives):3} active, {len(dones):3} complete')
                if good_dones:
                    best_metric = min(e.metric for e in good_dones)
                    best_trial = [e.trial for e in dones if e.metric == best_metric][0]
                    to_print += f', {best_metric:10.4f} best {best_trial}'
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
        requested_round = self.brackets[bracket_idx][round_idx]
        requested_element = None
        for el in requested_round:
            if el.trial == trial:
                requested_element = el
        assert requested_element
        requested_element.metric = metric
        requested_element.active = False
        if math.isfinite(metric):
            log_print('hyperband received report', bracket_idx, round_idx, trial, metric)
        else:
            log_print(Fore.LIGHTRED_EX + 'hyperband received report', bracket_idx, round_idx, trial, metric)
            # reset first rounder null results
            if round_idx == 0:
                requested_element.trial = None
                requested_element.metric = None


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
            Fore.LIGHTBLUE_EX + f'amended trial {old_trial} with budget '
                                f'{new_state["start_budget"]} -> {new_state["end_budget"]}, params',
            new_state['params'])

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
@click.option('--exp-path', required=True)
@click.option('--module', required=True)
@click.option('--min-budget', type=int, default=1)
@click.option('--max-budget', type=int, required=True)
@click.option('--sleep-interval', type=float, default=10.)
@click.option('--max-hyperbands', type=int, default=2)
@click.option('--reduction-ratio', type=float, default=math.e)
@click.option('--mode', type=click.Choice(['maximize', 'minimize']), default='minimize')
@click.option('--bohb-random-ratio', type=float, default=0.15)
@click.option('--bohb-guided-ratio', type=float, default=0.7)
@click.option('--bohb-random-size', type=int, default=64)
@click.option('--bohb-guided-size', type=int, default=128)
@click.option('--bohb-result-size-threshold', type=int)
@click.option('--bohb-good-ratio', type=float, default=0.15)
@click.option('--bohb-model-cache-time', type=float, default=30.)
@click.option('--bohb-min-bandwidth', type=float, default=1e-3)
@click.option('--bohb-bandwidth-estimation', default='normal_reference')
@click.option('--bohb-bandwidth-factor', type=float, default=3)
def run(module, exp_path, min_budget, max_budget, reduction_ratio, max_hyperbands, sleep_interval, mode,
        bohb_random_ratio, bohb_guided_ratio, bohb_random_size, bohb_guided_size,
        bohb_result_size_threshold, bohb_good_ratio, bohb_model_cache_time,
        bohb_min_bandwidth, bohb_bandwidth_estimation, bohb_bandwidth_factor):
    exp_path = os.path.abspath(exp_path)
    trial_gen = Generator(module, exp_path)
    param_gen = BOHBParamGen(module, exp_path,
                             random_ratio=bohb_random_ratio,
                             random_sample_size=bohb_random_size,
                             guided_ratio=bohb_guided_ratio,
                             guided_sample_size=bohb_guided_size,
                             result_size_threshold=bohb_result_size_threshold,
                             good_ratio=bohb_good_ratio,
                             model_cache_time=bohb_model_cache_time,
                             mode=mode,
                             min_bandwidth=bohb_min_bandwidth,
                             bandwidth_estimation=bohb_bandwidth_estimation,
                             bandwidth_factor=bohb_bandwidth_factor)
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
