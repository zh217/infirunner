import functools
import inspect
import numpy as np
import random

from abc import ABC, abstractmethod
from box import Box


class Param(ABC):
    def __init__(self, capsule, key, default):
        self._capsule = capsule
        self._key = key
        self.default = default

    @abstractmethod
    def get_next_value(self):
        pass

    @abstractmethod
    def serialize_as_nni(self):
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
                if not self._capsule.initialized:
                    raise RuntimeError('Must initialize capsule before use')
                params = self._process_params(params)
                return f(this, params, **kwargs)
        else:
            @functools.wraps(f)
            def wrapper(params=None, **kwargs):
                if not self._capsule.initialized:
                    raise RuntimeError('Must initialize capsule before use')
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

    def serialize_as_nni(self):
        return None


class ChoiceParam(Param):
    def __init__(self, capsule, key, default, choices, prob=None):
        super().__init__(capsule, key, default)
        self.choices = choices
        self.prob = prob

    def get_next_value(self):
        return np.random.choice(self.choices, p=self.prob)

    def serialize_as_nni(self):
        return {
            '_type': 'choice',
            '_value': self.choices
        }


class OrderedParam(Param):
    def __init__(self, capsule, key, default, choices, prob=None):
        super().__init__(capsule, key, default)
        self.choices = choices
        self.prob = prob

    def get_next_value(self):
        return np.random.choice(self.choices, p=self.prob)

    def serialize_as_nni(self):
        return {
            '_type': 'choice',
            '_value': self.choices
        }

class RandomIntParam(Param):
    def __init__(self, capsule, key, default, low, high):
        super().__init__(capsule, key, default)
        self.low = low
        self.high = high

    def get_next_value(self):
        return np.random.randint(self.low, self.high)

    def serialize_as_nni(self):
        return {
            '_type': 'randint',
            '_value': [self.low, self.high]
        }


class UniformParam(Param):
    def __init__(self, capsule, key, default, low, high):
        super().__init__(capsule, key, default)
        self.low = low
        self.high = high

    def get_next_value(self):
        return np.random.uniform(self.low, self.high)

    def serialize_as_nni(self):
        return {
            '_type': 'uniform',
            '_value': [self.low, self.high]
        }


def _clip(val, q, low, high):
    return np.clip(np.round(val / q) * q, low, high)


class QUniformParam(Param):
    def __init__(self, capsule, key, default, low, high, q):
        super().__init__(capsule, key, default)
        self.low = low
        self.high = high
        self.q = q

    def get_next_value(self):
        return _clip(random.uniform(self.low, self.high), self.q, self.low, self.high)

    def serialize_as_nni(self):
        return {
            '_type': 'quniform',
            '_value': [self.low, self.high, self.q]
        }


class LogUniformParam(Param):
    def __init__(self, capsule, key, default, low, high):
        super().__init__(capsule, key, default)
        self.low = low
        self.high = high

    def get_next_value(self):
        return np.exp(np.random.uniform(np.log(self.low), np.log(self.high)))

    def serialize_as_nni(self):
        return {
            '_type': 'loguniform',
            '_value': [self.low, self.high]
        }


class QLogUniformParam(Param):
    def __init__(self, capsule, key, default, low, high, q):
        super().__init__(capsule, key, default)
        self.low = low
        self.high = high
        self.q = q

    def get_next_value(self):
        return _clip(np.exp(np.random.uniform(np.log(self.low), np.log(self.high))), self.q, self.low, self.high)

    def serialize_as_nni(self):
        return {
            '_type': 'qloguniform',
            '_value': [self.low, self.high, self.q]
        }


class NormalParam(Param):
    def __init__(self, capsule, key, default, mean, std):
        super().__init__(capsule, key, default)
        self.mean = mean
        self.std = std

    def get_next_value(self):
        return np.random.normal(self.mean, self.std)

    def serialize_as_nni(self):
        return {
            '_type': 'normal',
            '_value': [self.mean, self.std]
        }


class QNormalParam(Param):
    def __init__(self, capsule, key, default, mean, std, q):
        super().__init__(capsule, key, default)
        self.mean = mean
        self.std = std
        self.q = q

    def get_next_value(self):
        return np.round(np.random.normal(self.mean, self.std) / self.q) * self.q

    def serialize_as_nni(self):
        return {
            '_type': 'qnormal',
            '_value': [self.mean, self.std, self.q]
        }


class LogNormalParam(Param):
    def __init__(self, capsule, key, default, mean, std):
        super().__init__(capsule, key, default)
        self.mean = mean
        self.std = std

    def get_next_value(self):
        return np.exp(np.random.normal(self.mean, self.std))

    def serialize_as_nni(self):
        return {
            '_type': 'lognormal',
            '_value': [self.mean, self.std]
        }


class QLogNormalParam(Param):
    def __init__(self, capsule, key, default, mean, std, q):
        super().__init__(capsule, key, default)
        self.mean = mean
        self.std = std
        self.q = q

    def get_next_value(self):
        return np.round(np.exp(np.random.normal(self.mean, self.std)) / self.q) * self.q

    def serialize_as_nni(self):
        return {
            '_type': 'qlognormal',
            '_value': [self.mean, self.std, self.q]
        }


class ParamGenerator:
    def __init__(self, capsule):
        self.capsule = capsule

    def const(self, key, default, value):
        return self._make_param_wrapper(key, ConstParam(self.capsule, key, default, value))

    def choice(self, key, default, choices, prob=None):
        return self._make_param_wrapper(key, ChoiceParam(self.capsule, key, default, choices, prob))

    def ordered(self, key, default, choices, prob=None):
        return self._make_param_wrapper(key, OrderedParam(self.capsule, key, default, choices, prob))

    def randint(self, key, default, low, high):
        return self._make_param_wrapper(key, RandomIntParam(self.capsule, key, default, low, high))

    def unif(self, key, default, low, high):
        return self._make_param_wrapper(key, UniformParam(self.capsule, key, default, low, high))

    def qunif(self, key, default, low, high, q):
        return self._make_param_wrapper(key, QUniformParam(self.capsule, key, default, low, high, q))

    def logunif(self, key, default, low, high):
        return self._make_param_wrapper(key, LogUniformParam(self.capsule, key, default, low, high))

    def qlogunif(self, key, default, low, high, q):
        return self._make_param_wrapper(key, QLogUniformParam(self.capsule, key, default, low, high, q))

    def normal(self, key, default, mean, std):
        return self._make_param_wrapper(key, NormalParam(self.capsule, key, default, mean, std))

    def qnormal(self, key, default, mean, std, q):
        return self._make_param_wrapper(key, QNormalParam(self.capsule, key, default, mean, std, q))

    def lognormal(self, key, default, mean, std):
        return self._make_param_wrapper(key, LogNormalParam(self.capsule, key, default, mean, std))

    def qlognormal(self, key, default, mean, std, q):
        return self._make_param_wrapper(key, QLogNormalParam(self.capsule, key, default, mean, std, q))

    def _make_param_wrapper(self, key, param_wrapper):
        self.capsule._guard_params()
        if key in self.capsule._param_wrappers:
            raise ValueError('duplicate parameter key', key)
        self.capsule._param_wrappers[key] = param_wrapper
        return param_wrapper