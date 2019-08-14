import abc
import enum
import importlib
import math
import os
import sys
import time

import click
import infirunner.capsule

import infirunner.param as param
from collections import namedtuple

from infirunner.util import make_trial_id
from infirunner.generator import Generator
from infirunner.watch import ExperimentWatcher


def int_ceil(x):
    return int(math.ceil(x))


def int_floor(x):
    return int(math.floor(x))


class ElementVerdict(enum.Enum):
    GOOD = 'good'
    BAD = 'bad'
    UNKNOWN = 'unknown'


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

    def reset_model(self, good_elements, bad_elements):
        pass

    def get_next_parameter(self):
        return self.capsule.gen_params(use_default=False, skip_const=True)


class RandomParamGen(ParamGen):
    pass


class BOHBParamGen(ParamGen):
    def __init__(self, module, n_trials):
        super().__init__(module)
        self.n_trials = n_trials

    def reset_model(self, good_elements, bad_elements):
        pass

    def get_next_parameter(self):
        if True:
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
            print(f'bracket {bracket_idx}')
            for round_idx, round in enumerate(bracket):
                actives = [e for e in round if e.active]
                dones = [e for e in round if e.metric is not None]
                budget = round[0].budget
                print(
                    f'\tround {round_idx:1} each with {budget:4} budgets, '
                    f'{len(round):4} trials, {len(actives):4} active, {len(dones):4} complete')

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

    def request_trial(self):
        # if all brackets are complete, raise StopIteration
        # if caller should wait, return None
        # return: BracketElement, note that if el.trial is empty the caller is responsible for filling it
        if self.cur_bracket_idx > self.bracket_max:
            raise StopIteration
        cur_bracket = self.brackets[self.cur_bracket_idx]
        cur_round = cur_bracket[self.cur_round_idx]
        # with_metrics = [e for e in cur_round if e.metric is not None]

        inactive_without_trial = [e for e in cur_round if e.trial is None and not e.active]
        if inactive_without_trial:
            ret = inactive_without_trial[0]

            if self.cur_round_idx == 0:
                ret.active = True
                return ret
            else:
                # if not self.is_round_complete(self.cur_bracket_idx, self.cur_round_idx - 1):
                #     return None

                cur_round_trial_ids = set(e.trial for e in cur_round if e.trial is not None)
                last_round_completed_trials = [e for e in cur_bracket[self.cur_round_idx - 1]
                                               if e.metric is not None and math.isfinite(e.metric)
                                               and e.trial not in cur_round_trial_ids]
                if last_round_completed_trials:
                    # this COULD be empty, since all elements in previous round may have failed with NaN
                    # in that case the caller should initialize a new one to run
                    last_round_completed_trials.sort(key=lambda e: e.metric)
                    ret.trial = last_round_completed_trials[0].trial

                ret.active = True
                # if no trial is present, the caller is responsible for filling it it
                return ret

        # if any(e.trial is None and e.active for e in cur_round):
        #     return None

        # change idx and recur

        if self.is_round_complete(self.cur_bracket_idx, self.cur_round_idx):
            if self.cur_round_idx == len(cur_bracket) - 1:
                self.cur_round_idx = 0
                self.cur_bracket_idx += 1
            else:
                self.cur_round_idx += 1
            print(id(self), 'proceed to bracket', self.cur_bracket_idx, 'round', self.cur_round_idx)
            return self.request_trial()
        else:
            return None

    def report_trial(self, bracket_idx, round_idx, trial, metric):
        # mark inactive, set metric
        # make verdict for all completed rounds
        print('hyperband received report', bracket_idx, round_idx, trial, metric)
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
                 reduction_ratio, sleep_interval, max_hyperbands):
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

    def generate_new_trial(self, end_budget, n_gpu=1):
        params = self.param_generator.get_next_parameter()
        new_id = make_trial_id()
        print(f'generate new trial {new_id} with budget 0 -> {end_budget}')
        self.trial_generator.change_capsule_trial_id(new_id)
        self.trial_generator.save_start_state(start_budget=0,
                                              end_budget=end_budget,
                                              n_gpu=n_gpu,
                                              params=params)
        return new_id

    def amend_trial(self, old_trial, end_budget, n_gpu=1):
        self.trial_generator.change_capsule_trial_id(old_trial)
        new_state = self.trial_generator.amend_start_state(end_budget=end_budget, n_gpu=n_gpu)
        print(f'amended trial {old_trial} with budget {new_state["start_budget"]} -> {new_state["end_budget"]}')

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
            print('generated new hyperband group')
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
                print(f'obtained watcher result for {trial.trial}')
                if trial_result['budget'] != trial.budget:
                    trial_result['metric'] = float('nan')
                if trial_result['metric'] is None:
                    trial_result['metric'] = float('nan')
                self.hyperbands[hyperband_idx].report_trial(trial.bracket, trial.round, trial.trial,
                                                            trial_result['metric'])
        self.watch_active_trials[:] = [t for t in self.watch_active_trials if t[1].trial not in completed_trials]

    def start_trials(self):
        n_slots = self.get_available_slots()
        for _ in range(n_slots):
            hyperband_idx, new_trial = self.get_next_hyperband_trial()
            if new_trial is not None:
                # launch new trial
                # add to trials being watched
                print(f'watching trial {new_trial.trial} of band {hyperband_idx}')
                self.watch_active_trials.append((hyperband_idx, new_trial))

    def get_trial_current_budget(self, trial_id):
        pass

    def start(self):
        last_watching = set()
        while True:
            self.check_for_completed_trials()
            self.start_trials()
            cur_watching = set(t.trial for _, t in self.watch_active_trials)
            if last_watching != cur_watching:
                for idx, hb in enumerate(self.hyperbands):
                    print('----- Hyperband', idx, id(hb), '-----')
                    hb.pprint_brackets()
            last_watching = cur_watching
            time.sleep(self.sleep_interval)


@click.command()
@click.option('--exp-path', default='_exp')
@click.option('--module', required=True)
@click.option('--min-budget', type=int, default=1)
@click.option('--max-budget', type=int, required=True)
@click.option('--sleep-interval', type=float, default=10.)
@click.option('--max-hyperbands', type=int, default=2)
@click.option('--reduction-ratio', type=float, default=math.e)
def run(module, exp_path, min_budget, max_budget, reduction_ratio, max_hyperbands, sleep_interval):
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
                             max_hyperbands=max_hyperbands)
    driver.start()


if __name__ == '__main__':
    run()
