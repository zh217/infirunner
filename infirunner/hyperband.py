import abc
import enum
import math
import infirunner.param as param
from collections import namedtuple


def int_ceil(x):
    return int(math.ceil(x))


def int_floor(x):
    return int(math.floor(x))


class ElementVerdict(enum.Enum):
    GOOD = 'good'
    BAD = 'bad'
    UNKNOWN = 'unknown'


# Note that the metric is always assumed to be optimized towards minimum.
BracketElement = namedtuple('BracketElement', ['bracket', 'round', 'budget', 'metric', 'trial', 'active'])


class ParamGen(abc.ABC):
    def __init__(self, capsule):
        self.capsule = capsule

    def reset_model(self, good_elements, bad_elements):
        pass

    def get_next_parameter(self):
        return self.capsule.gen_params(use_default=False, skip_const=True)


class RandomParamGen(ParamGen):
    pass


class BOHBParamGen(ParamGen):
    def __init__(self, capsule, n_trials):
        super().__init__(capsule)
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
                budget = round[0].budget
                print(
                    f'\tround {round_idx:1} each with {budget:4} budgets, '
                    f'{len(round):4} trials, {len(actives):4} active')

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

    def request_trial(self):
        # if all brackets are complete, raise StopIteration
        # if caller should wait, return None
        # return: BracketElement, note that if el.trial is empty the caller is responsible for filling it
        if self.cur_bracket_idx >= self.bracket_max:
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
                if not all(e.metric is not None for e in cur_bracket[self.cur_round_idx - 1]):
                    # should wait for previous round to finish first
                    return None

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

        if any(e.trial is None and e.active for e in cur_round):
            return None

        # change idx and recur
        if self.cur_round_idx == len(cur_bracket) - 1:
            self.cur_round_idx = 0
            self.cur_bracket_idx += 1
        else:
            self.cur_round_idx += 1
        return self.request_trial()

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


class HyperbandDriver:
    def __init__(self, experiment_dir, min_budget, max_budget, reduction_ratio=math.e):
        self.experiment_dir = experiment_dir
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.reduction_ratio = reduction_ratio
        self.hyperbands = []

    def generate_new_trial(self):
        return 'xx'

    def get_next_hyperband_trial(self):
        for hyperband in self.hyperbands:
            try:
                new_trial = hyperband.request_trial()
            except StopIteration:
                continue
            if new_trial is not None:
                if new_trial.trial is None:
                    new_trial.trial = self.generate_new_trial()
                return new_trial
        self.hyperbands.append(Hyperband(min_budget=self.min_budget,
                                         max_budget=self.max_budget,
                                         reduction_ratio=self.reduction_ratio))
        return self.get_next_hyperband_trial()

    def start_epoch(self):
        pass

    def get_trial_current_budget(self, trial_id):
        pass




if __name__ == '__main__':
    hb = Hyperband(min_budget=10,
                   max_budget=1000)
    hb.pprint_brackets()
