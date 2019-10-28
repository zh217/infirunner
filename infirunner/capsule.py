import functools
import math
import os
import random
import time
import json
import sys
import inspect
import shutil

import torch
import infirunner.steppers
import infirunner.param

import numpy as np

from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter

from infirunner.util import make_trial_id

DEBUG_MODE = 'debug'
TRAIN_MODE = 'train'
TURBO_MODE = 'turbo'


class RunnerCapsule:
    def __init__(self, mode, turbo_index, exp_path, trial_id, budget_start, budget_end, start_params):
        self.param = infirunner.param.ParamGenerator(self)
        self.params = start_params or {}
        self.turbo_index = turbo_index
        self.initialized = False
        self.exp_path = os.path.abspath(exp_path)
        self.trial_id = trial_id
        self._param_wrappers = {}
        self._tb_writer = None
        self.steps = 0
        self.log_files = {}
        self.get_model_state = None
        self.get_metadata_state = None
        self.stdout = self.stderr = self.orig_stdout = self.orig_stderr = None
        self.mode = self.set_mode(mode)
        self.prev_time = 0
        self.start_time = time.time()
        self.budget_current = self.budget_start = budget_start
        self.budget_end = budget_end
        self.metric = None

    @property
    def var_params(self):
        res = {}
        for k, v in self.param_gen.items():
            if not isinstance(v, infirunner.param.ConstParam):
                res[k] = self.params[k]
        return res

    @property
    def param_gen(self):
        return self._param_wrappers

    @property
    def save_path(self):
        return os.path.join(self.exp_path, self.trial_id)

    def is_leader(self):
        return self.turbo_index == 0

    def is_debug(self):
        return self.mode == DEBUG_MODE

    def is_train(self):
        return self.mode == TRAIN_MODE

    def is_turbo(self):
        return self.mode == TURBO_MODE

    def set_mode(self, mode):
        assert mode in (DEBUG_MODE, TRAIN_MODE, TURBO_MODE)
        self.mode = mode
        return mode

    def redirect_io(self):
        os.makedirs(self.save_path, exist_ok=True)
        stdout_file = os.path.join(self.save_path, f'std_out_{self.turbo_index}.log')
        stderr_file = os.path.join(self.save_path, f'std_err_{self.turbo_index}.log')
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        self.stdout = sys.stdout = open(stdout_file, 'a', encoding='utf-8')
        self.stderr = sys.stderr = open(stderr_file, 'a', encoding='utf-8')

    def restore_io(self):
        if self.stdout:
            sys.stdout = self.orig_stdout
        if self.stderr:
            sys.stderr = self.orig_stderr
        try:
            self.stdout.close()
        except:
            pass
        try:
            self.stderr.close()
        except:
            pass
        del self.stdout
        del self.stderr

    def save_sources(self, white_list=None):
        if not self.is_leader():
            return

        if white_list is None:
            frm = inspect.stack()[1]
            mod = inspect.getmodule(frm[0])
            if not mod:
                raise AttributeError('save_sources needs to be called from a module!')
            white_list = [mod.__name__.split('.')[0]]

        source_dir = os.path.join(self.save_path, 'src')
        if os.path.exists(source_dir):
            shutil.rmtree(source_dir, ignore_errors=True)
        mods = {k: m for k, m in sys.modules.items()
                if any(k.startswith(p) for p in white_list)}

        for k, m in mods.items():
            try:
                source = inspect.getsource(m)
            except OSError:
                continue
            except TypeError:
                continue
            m_comps = k.split('.')
            source_save_path = os.path.join(source_dir, *m_comps[:-1])
            os.makedirs(source_save_path, exist_ok=True)
            with open(os.path.join(source_save_path, m_comps[-1] + '.py'), 'w', encoding='utf-8') as f:
                f.write(source)

    def set_state_getter(self, model_state_getter, metadata_state_getter):
        self.get_model_state = model_state_getter
        self.get_metadata_state = metadata_state_getter

    def serialize_state(self):
        now = time.time()
        return {
            'budget_start': self.budget_start,
            'budget_end': self.budget_end,
            'budget_current': self.budget_current,
            'save_path': self.save_path,
            'trial_id': self.trial_id,
            'prev_time': self.prev_time,
            'start_time': self.start_time,
            'cur_time': now,
            'relative_time': self.prev_time + now - self.start_time,
            'mode': self.mode,
            'steps': self.steps,
            'metric': self.metric,
            'params': self.params,
            'param_gens': self.serialize_param_gen()
        }

    def running_average(self, key):
        return infirunner.steppers.RunningAverage(self, key)

    def running_averages(self, *keys, prefix=''):
        return infirunner.steppers.RunningAverageGroup(self, keys, prefix=prefix)

    def step(self, size=1):
        self.steps += size

    def __del__(self):
        for file in self.log_files.values():
            try:
                file.close()
            except:
                pass

    def get_log_file_handle(self, key):
        log_file = self.log_files.get(key)
        if log_file is None:
            os.makedirs(os.path.join(self.save_path, 'logs'), exist_ok=True)
            log_file = open(os.path.join(self.save_path, 'logs', f'{key}.tsv'), 'a', encoding='utf-8')
        return log_file

    def log_scalar(self, key, value):
        if not self.is_leader():
            return
        now = time.time()
        rel_time = self.prev_time + now - self.start_time
        self.get_log_file_handle(key).write(f'{self.steps}\t{now}\t{rel_time}\t{value}\n')

    def log_scalars(self, key, values):
        if not self.is_leader():
            return
        log_file = self.get_log_file_handle(key)
        now = time.time()
        rel_time = self.prev_time + now - self.start_time
        for v in values:
            log_file.write(f'{self.steps}\t{now}\t{rel_time}\t{v}\n')

    def log_file(self, key, ext, data):
        if not self.is_leader():
            return
        p = os.path.join(self.save_path, 'logs', key)
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, f'{self.steps:015}.{ext}', 'wb')) as file:
            file.write(data)

    def serialize_param_gen(self):
        ret = {}
        for k, v in self._param_wrappers.items():
            ret[k] = {'type': v.__class__.__name__,
                      'opts': v.serialize()}
        return ret

    def _guard_params(self):
        if self.initialized:
            raise RuntimeError('Cannot produce/set parameters after it is used')

    def set_params(self, param_map):
        self._guard_params()
        self.params.update(param_map)

    def report_metric(self, metric, save_state=True, consume_budget=1):
        self.budget_current += consume_budget
        self.metric = metric
        with open(os.path.join(self.save_path, 'metric.tsv'), 'a', encoding='ascii') as f:
            f.write(f'{self.budget_current}\t{time.time()}\t{metric}\n')
        if save_state:
            self.save_state()
        if not math.isfinite(metric) or self.budget_current >= self.budget_end:
            if self._tb_writer is not None:
                self._tb_writer.close()
            print('exit at metric', metric, 'budget', self.budget_current, file=sys.stderr)
            sys.exit()

    def save_state(self, empty_ok=False):
        if self.turbo_index != 0:
            return
        out_dir = os.path.join(self.save_path, 'saves', f'{self.budget_current:05}')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'state.json'), 'w', encoding='utf-8') as f:
            json.dump(self.serialize_state(), f, ensure_ascii=False, indent=2, allow_nan=True)
        if self.get_metadata_state:
            with open(os.path.join(out_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(self.get_metadata_state(), f, ensure_ascii=False, indent=2, allow_nan=True)
        if not empty_ok:
            assert self.get_model_state
        if self.get_model_state:
            torch.save(self.get_model_state(), os.path.join(out_dir, 'model.pt'))

    def load_state(self, load_path, override_params=True):
        if load_path is None:
            return None, None
        with open(os.path.join(load_path, 'state.json'), 'r', encoding='utf-8') as f:
            meta = json.load(f)
            self.steps = meta['steps']
            self.prev_time = meta['relative_time']
            print('loading saved states from', load_path, 'steps', self.steps, 'prev_time', self.prev_time)
            if override_params:
                self.params = meta['params']

        try:
            with open(os.path.join(load_path, 'metadata.json'), 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = None

        try:
            model_state = torch.load(os.path.join(load_path, 'model.pt'), map_location='cpu')
        except FileNotFoundError:
            model_state = None
        return model_state, metadata

    def get_tb_writer(self):
        if not self.is_leader():
            return None
        if self._tb_writer is None:
            self._tb_writer = SummaryWriter(log_dir=self.save_path)
        return self._tb_writer

    def gen_params(self, use_default=False, skip_const=False):
        new_params = {}
        for k, w in self._param_wrappers.items():
            if skip_const and type(w) == infirunner.param.ConstParam:
                continue
            new_params[k] = w.default if use_default else w.get_next_value()
        return new_params

    def initialize(self):
        if not self.initialized:
            self.initialized = True
            new_params = self.gen_params(use_default=self.mode == DEBUG_MODE, skip_const=False)
            new_params.update(self.params)
            self.params = new_params

    @contextmanager
    def deterministically_stochastic(self):
        old_cuda_state, old_np_state, old_state, old_th_state = self.seed_random()

        yield

        if torch.cuda.is_available():
            torch.cuda.set_rng_state(old_cuda_state)
        torch.set_rng_state(old_th_state)
        np.random.set_state(old_np_state)
        random.setstate(old_state)

    def seed_random(self):
        old_state = random.getstate()
        old_np_state = np.random.get_state()
        old_th_state = torch.get_rng_state()
        if torch.cuda.is_available():
            old_cuda_state = torch.cuda.get_rng_state()
        else:
            old_cuda_state = None
        seed = self.turbo_index + 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        return old_cuda_state, old_np_state, old_state, old_th_state

    def load(self, load_budget=None, initialize=True):
        if self.initialized:
            raise RuntimeError('Cannot call load after initialization')
        if load_budget is None:
            load_budget = self.budget_start
        if load_budget > 0:
            load_path = os.path.join(self.save_path, 'saves', f'{load_budget:05}')
        else:
            load_path = None
        if initialize:
            self.initialize()
        return self.load_state(load_path)


class DynamicStateGetter:
    def __init__(self):
        self.state_getters = set()

    def add_state_getter(self, new_state_getter):
        self.state_getters.add(new_state_getter)

    def remove_state_getter(self, state_getter):
        self.state_getters.remove(state_getter)

    def __call__(self):
        ret = {}
        for state_getter in self.state_getters:
            ret.update(state_getter())
        return ret


active_capsule = None


def make_capsule():
    exp_path = os.environ.get('INFR_EXP_PATH')
    if exp_path is None:
        exp_path = os.path.join(os.getcwd(), '_infirunner')

    mode = os.environ.get('INFR_MODE')
    assert mode is None or mode in (DEBUG_MODE, TRAIN_MODE, TURBO_MODE)
    if mode is None:
        mode = DEBUG_MODE
    turbo_index = int(os.environ.get('INFR_TURBO_INDEX', '0'))
    trial_id = os.environ.get('INFR_TRIAL')
    if trial_id is None:
        trial_id = make_trial_id()
    start_state_str = os.environ.get('INFR_START_STATE')
    if start_state_str:
        with open(start_state_str, 'r', encoding='utf-8') as f:
            start_obj = json.load(f)
            budget_start = start_obj['start_budget']
            budget_end = start_obj['end_budget']
            start_params = start_obj['params']
    else:
        budget_start = 0
        budget_end = sys.maxsize
        start_params = {}
    budget_str = os.environ.get('INFR_BUDGET')
    if budget_str is not None:
        budget_start, budget_end = budget_str.split(',')
        budget_start = int(budget_start)
        budget_end = int(budget_end)
    _cap = RunnerCapsule(mode, turbo_index, exp_path, trial_id, budget_start, budget_end, start_params)
    if os.environ.get('INFR_REDIRECT_IO'):
        _cap.redirect_io()
    global active_capsule
    active_capsule = _cap
    return _cap


def chain(*decorators):
    def wrap_f(f):
        for dec in reversed(decorators):
            f = dec(f)
        return f

    return wrap_f
