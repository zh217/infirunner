import functools
import os
import random
import string
import time
import datetime
import json
import sys
import inspect
import shutil

import torch

import numpy as np

from contextlib import contextmanager
from abc import ABC, abstractmethod
from box import Box
from torch.utils.tensorboard import SummaryWriter


class RunningAverage:
    def __init__(self, capsule, key):
        self.key = key
        self.capsule = capsule
        self.count = 0
        self.sum = 0

    def update(self, *values):
        for value in values:
            self.count += 1
            self.sum += value

    def get(self):
        if self.count == 0:
            return 0
        else:
            return self.sum / self.count

    def reset(self):
        avg = self.get()
        self.count = 0
        self.sum = 0
        return avg

    def write_to_log(self, flush=False):
        self.capsule.log_scalar(self.key, self.get())
        if flush:
            self.reset()

    def write_to_tb(self, flush=True):
        writer = self.capsule.get_tb_writer()
        writer.add_scalar(self.key, self.get(), self.capsule.steps)
        if flush:
            self.reset()


class Timer:
    def __init__(self, timeout):
        self.last_time = time.time()
        self.timeout = timeout

    def reset(self):
        self.last_time = time.time()

    def ticked(self):
        now = time.time()
        if now - self.last_time > self.timeout:
            self.last_time = now
            return True
        else:
            return False


class StepCounter:
    def __init__(self, steps):
        self.steps = steps
        self.n = 0

    def reset(self):
        self.n = 0

    def ticked(self):
        self.n += 1
        if self.n == self.steps:
            self.n = 0
            return True
        else:
            return False


class Param(ABC):
    def __init__(self, capsule, key, default):
        self._capsule = capsule
        self._key = key
        self.default = default

    @abstractmethod
    def get_next_value(self):
        pass

    def serialize(self):
        ret = {}
        for k, v in self.__dict__.items():
            if k and k[0] != '_':
                ret[k] = v
        return ret

    def __call__(self, f):
        if inspect.ismethod(f) or f.__name__ == '__init__':
            @functools.wraps(f)
            def wrapper(this, params=None, **kwargs):
                self._capsule.mark_used()
                params = self._process_params(params)
                return f(this, params, **kwargs)
        else:
            @functools.wraps(f)
            def wrapper(params=None, **kwargs):
                self._capsule.mark_used()
                params = self._process_params(params)

                return f(params, **kwargs)

        return wrapper

    def _process_params(self, params):
        _params = params
        if not isinstance(_params, Box):
            params = Box(default_box=True)
        else:
            params = _params
        if _params is not None:
            for ks, v in _params.items():
                ks = ks.split('.')
                cur = params
                for k in ks[:-1]:
                    cur = cur[k]
                cur[ks[-1]] = v
        cur = params
        ks = self._key.split('.')
        for k in ks[:-1]:
            cur = cur[k]
        if ks[-1] not in cur:
            cur[ks[-1]] = self._capsule.params[self._key]
        return params

    def __repr__(self):
        repr_str = f'<{self.__class__.__name__}'
        for k, v in self.serialize().items():
            repr_str += f' {k}={repr(v)}'
        repr_str += '>'
        return repr_str


class ConstParam(Param):
    def __init__(self, capsule, key, default, const):
        super().__init__(capsule, key, default)
        self.const = const

    def get_next_value(self):
        return self.const


class ChoiceParam(Param):
    def __init__(self, capsule, key, default, choices, prob=None):
        super().__init__(capsule, key, default)
        self.choices = choices
        self.prob = prob

    def get_next_value(self):
        return np.random.choice(self.choices, p=self.prob)


class RandomIntParam(Param):
    def __init__(self, capsule, key, default, low, high):
        super().__init__(capsule, key, default)
        self.low = low
        self.high = high

    def get_next_value(self):
        return np.random.randint(self.low, self.high)


class UniformParam(Param):
    def __init__(self, capsule, key, default, low, high):
        super().__init__(capsule, key, default)
        self.low = low
        self.high = high

    def get_next_value(self):
        return np.random.uniform(self.low, self.high)


class LogUniformParam(Param):
    def __init__(self, capsule, key, default, low, high):
        super().__init__(capsule, key, default)
        self.low = low
        self.high = high

    def get_next_value(self):
        return np.exp(np.random.uniform(np.log(self.low), np.log(self.high)))


class NormalParam(Param):
    def __init__(self, capsule, key, default, mean, std):
        super().__init__(capsule, key, default)
        self.mean = mean
        self.std = std

    def get_next_value(self):
        return np.random.normal(self.mean, self.std)


DEBUG_MODE = 'debug_mode'
SEARCH_MODE = 'search_mode'
TURBO_MODE = 'turbo_mode'


class RunnerCapsule:
    def __init__(self, mode, turbo_index, save_path, trial_id, steps):
        self.params = {}
        self.turbo_index = turbo_index
        self.initialized = False
        self.save_path = save_path
        self.trial_id = trial_id
        self._param_wrappers = {}
        self._tb_writer = None
        self.steps = steps
        self.log_files = {}
        self.metric = None
        self.get_model_state = None
        self.get_metadata_state = None
        self.stdout = self.stderr = self.orig_stdout = self.orig_stderr = None
        self.mode = self.set_mode(mode)

    def is_debug(self):
        return self.mode == DEBUG_MODE

    def is_search(self):
        return self.mode == SEARCH_MODE

    def is_turbo(self):
        return self.mode == TURBO_MODE

    def set_mode(self, mode):
        assert mode in (DEBUG_MODE, SEARCH_MODE, TURBO_MODE)
        self.mode = mode
        if self.mode == DEBUG_MODE:
            self.restore_io()
        else:
            self.redirect_io()
        return mode

    def redirect_io(self):
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
        self.load()
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
            m_comps = k.split('.')
            source_save_path = os.path.join(source_dir, *m_comps[:-1])
            os.makedirs(source_save_path, exist_ok=True)
            with open(os.path.join(source_save_path, m_comps[-1] + '.py'), 'w', encoding='utf-8') as f:
                f.write(source)

        with open(os.path.join(source_dir, 'state.json'), 'w') as f:
            json.dump(self.serialize_state(), f, ensure_ascii=False, indent=2, allow_nan=True)

    def set_state_getter(self, model_state_getter, metadata_state_getter):
        self.get_model_state = model_state_getter
        self.get_metadata_state = metadata_state_getter

    def serialize_state(self):
        return {
            'save_path': self.save_path,
            'trial_id': self.trial_id,
            'mode': self.mode,
            'metric': self.metric,
            'steps': self.steps,
            'params': self.params,
            'param_gens': self.serialize_param_gen()
        }

    def running_average(self, key):
        return RunningAverage(self, key)

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
            log_file = open(os.path.join(self.save_path, 'logs', f'{key}.tsv'), 'a')
        return log_file

    def log_scalar(self, key, value):
        self.get_log_file_handle(key).write(f'{self.steps}\t{value}\n')

    def log_scalars(self, key, values):
        log_file = self.get_log_file_handle(key)
        for v in values:
            log_file.write(f'{self.steps}\t{v}\n')

    def log_file(self, key, ext, data):
        p = os.path.join(self.save_path, 'logs', key)
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, f'{self.steps:015}.{ext}', 'wb')) as file:
            file.write(data)

    def const(self, key, default, value):
        return self._make_param_wrapper(key, ConstParam(self, key, default, value))

    def choice(self, key, default, choices, prob=None):
        return self._make_param_wrapper(key, ChoiceParam(self, key, default, choices, prob))

    def randint(self, key, default, low, high):
        return self._make_param_wrapper(key, RandomIntParam(self, key, default, low, high))

    def unif(self, key, default, low, high):
        return self._make_param_wrapper(key, UniformParam(self, key, default, low, high))

    def logunif(self, key, default, low, high):
        return self._make_param_wrapper(key, LogUniformParam(self, key, default, low, high))

    def normal(self, key, default, mean, std):
        return self._make_param_wrapper(key, LogUniformParam(self, key, default, mean, std))

    def serialize_param_gen(self):
        ret = {}
        for k, v in self._param_wrappers.items():
            ret[k] = {'type': v.__class__.__name__,
                      'opts': v.serialize()}
        return ret

    def _guard_params(self):
        if self.initialized:
            raise RuntimeError('Cannot produce/set parameters after it is used')

    def _make_param_wrapper(self, key, param_wrapper):
        self._guard_params()
        if key in self._param_wrappers:
            raise ValueError('duplicate parameter key', key)
        self._param_wrappers[key] = param_wrapper
        return param_wrapper

    def set_params(self, param_map):
        self._guard_params()
        self.params.update(param_map)

    def set_metric(self, metric):
        self.metric = metric

    def save_state(self):
        if self.turbo_index != 0:
            return
        if self.metric is None:
            metric = 0.
        else:
            metric = self.metric
        out_dir = os.path.join(self.save_path, 'saves', f'{self.steps:015}_{metric:.5f}')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'state.json'), 'w') as f:
            json.dump(self.serialize_state(), f, ensure_ascii=False, indent=2, allow_nan=True)
        if self.get_metadata_state:
            with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
                json.dump(self.get_metadata_state(), f, ensure_ascii=False, indent=2, allow_nan=True)
        if self.get_model_state:
            torch.save(self.get_model_state(), os.path.join(out_dir, 'model.pt'))

    def load_state(self, load_path):
        try:
            with open(os.path.join(load_path, 'state.json'), 'r') as f:
                meta = json.load(f)
                self.steps = meta['steps']
                self.metric = meta['metric']
                self.params = meta['params']
        except FileNotFoundError:
            pass

        try:
            with open(os.path.join(load_path, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = None

        try:
            model_state = torch.load(os.path.join(load_path, 'model.pt'))
        except FileNotFoundError:
            model_state = None
        return model_state, metadata

    def get_tb_writer(self):
        if self._tb_writer is None:
            self._tb_writer = SummaryWriter(log_dir=self.save_path)
        return self._tb_writer

    def mark_used(self):
        if not self.initialized:
            self.initialized = True
            for k, w in self._param_wrappers.items():
                if k not in self.params:
                    self.params[k] = w.default if self.mode == DEBUG_MODE else w.get_next_value()

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

    def load(self, load_path=None):
        if self.initialized:
            return
        if load_path is None:
            try:
                saves = os.listdir(os.path.join(self.save_path, 'saves'))
                if saves:
                    saves.sort()
                    load_path = os.path.join(self.save_path, 'saves', saves[-1])
            except FileNotFoundError:
                pass
        if load_path is not None:
            self.load_state(load_path)
        self.mark_used()


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


def make_capsule():
    exp_path = os.environ.get('INFIRUNNER_EXP_PATH')
    if exp_path is None:
        exp_path = os.path.join(os.getcwd(), '_infirunner')

    mode = os.environ.get('INFIRUNNER_MODE', DEBUG_MODE)
    assert mode in (DEBUG_MODE, SEARCH_MODE, TURBO_MODE)
    turbo_index = int(os.environ.get('INFIRUNNER_TURBO_INDEX', '0'))
    trial_id = os.environ.get('INFIRUNNER_TRIAL_ID', None)
    if trial_id is None:
        rd_id = ''.join(random.choice(string.ascii_letters) for _ in range(6))
        trial_id = f'{datetime.datetime.now():%y%m%d_%H%M%S}_{rd_id}'
    save_path = os.path.join(exp_path, trial_id)
    os.makedirs(save_path, exist_ok=True)
    steps = int(os.environ.get('INFIRUNNER_TURBO_INDEX', '0'))
    _cap = RunnerCapsule(mode, turbo_index, save_path, trial_id, steps)
    return _cap


runner = make_capsule()
