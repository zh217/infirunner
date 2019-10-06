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

    @property
    def var_type(self):
        return 'c'

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

    @abstractmethod
    def encode_as_numerical(self, value):
        pass

    @abstractmethod
    def encoded_bounds(self):
        pass

    @abstractmethod
    def decode_from_numerical(self, coded):
        pass

    def __repr__(self):
        repr_str = f'<{self.__class__.__name__}'
        for k, v in self.serialize().items():
            repr_str += f' {k}={repr(v)}'
        repr_str += '>'
        return repr_str


def _clip_to_q(val, q, low, high):
    return np.clip(np.round(val / q) * q, low, high)


class ConstParam(Param):
    def __init__(self, capsule, key, default, const):
        super().__init__(capsule, key, default)
        self.const = const

    @property
    def var_type(self):
        return 'u'

    def encode_as_numerical(self, value):
        return 0

    def decode_from_numerical(self, coded):
        return 0

    def encoded_bounds(self):
        return None, None

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
        return random.choices(self.choices, weights=self.prob, k=1)[0]

    @property
    def var_type(self):
        return 'u'

    def encode_as_numerical(self, value):
        try:
            return self.choices.index(value)
        except:
            raise ValueError('Cannot encode choice: given', value, 'expect', self.choices)

    def encoded_bounds(self):
        return None, None

    def decode_from_numerical(self, coded):
        rounded = int(_clip_to_q(coded, q=1, low=0, high=len(self.choices) - 1))
        return self.choices[rounded]

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
        return random.choices(self.choices, weights=self.prob, k=1)[0]

    @property
    def var_type(self):
        return 'o'

    def encode_as_numerical(self, value):
        return self.choices.index(value)

    def decode_from_numerical(self, coded):
        rounded = int(_clip_to_q(coded, q=1, low=0, high=len(self.choices) - 1))
        return self.choices[rounded]

    def encoded_bounds(self):
        return -0.5, len(self.choices) - 0.5

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

    def encode_as_numerical(self, value):
        return (value - self.low) / (self.high - self.low)

    def decode_from_numerical(self, coded):
        restored = int(_clip_to_q(coded * (self.high - self.low) + self.low, 1, self.low, self.high))
        return restored

    def encoded_bounds(self):
        return (self.low - 0.5) / (self.high - self.low), (self.high + 0.5) / (self.high - self.low)

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

    def encode_as_numerical(self, value):
        return (value - self.low) / (self.high - self.low)

    def decode_from_numerical(self, coded):
        restored = np.clip(coded * (self.high - self.low) + self.low, self.low, self.high)
        return restored

    def encoded_bounds(self):
        return 0., 1.

    def serialize_as_nni(self):
        return {
            '_type': 'uniform',
            '_value': [self.low, self.high]
        }


class QUniformParam(Param):
    def __init__(self, capsule, key, default, low, high, q):
        super().__init__(capsule, key, default)
        self.low = low
        self.high = high
        self.q = q

    def get_next_value(self):
        return _clip_to_q(random.uniform(self.low, self.high), self.q, self.low, self.high)

    def encode_as_numerical(self, value):
        return (value - self.low) / (self.high - self.low)

    def decode_from_numerical(self, coded):
        restored = coded * (self.high - self.low) + self.low
        return _clip_to_q(restored, self.q, self.low, self.high)

    def encoded_bounds(self):
        return 0., 1.

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

    def encode_as_numerical(self, value):
        return (np.log(value) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))

    def decode_from_numerical(self, coded):
        restored = np.exp(coded * (np.log(self.high) - np.log(self.low)) + np.log(self.low))
        return restored

    def encoded_bounds(self):
        return 0., 1.

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
        return _clip_to_q(np.exp(np.random.uniform(np.log(self.low), np.log(self.high))), self.q, self.low, self.high)

    def encode_as_numerical(self, value):
        return (np.log(value) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))

    def decode_from_numerical(self, coded):
        restored = np.exp(coded * (np.log(self.high) - np.log(self.low)) + np.log(self.low))
        return _clip_to_q(restored, self.q, self.low, self.high)

    def encoded_bounds(self):
        return 0., 1.

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

    def encode_as_numerical(self, value):
        return (value - self.mean) / self.std

    def decode_from_numerical(self, coded):
        restored = coded * self.std + self.mean
        return restored

    def encoded_bounds(self):
        return None, None

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

    def encode_as_numerical(self, value):
        return (value - self.mean) / self.std

    def decode_from_numerical(self, coded):
        restored = coded * self.std + self.mean
        return np.round(restored / self.q) * self.q

    def encoded_bounds(self):
        return None, None

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

    def encode_as_numerical(self, value):
        return (np.log(value) - self.mean) / self.std

    def decode_from_numerical(self, coded):
        restored = np.exp(coded * self.std + self.mean)
        return restored

    def encoded_bounds(self):
        return None, None

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

    def encode_as_numerical(self, value):
        return (np.log(value) - self.mean) / self.std

    def decode_from_numerical(self, coded):
        restored = np.exp(coded * self.std + self.mean)
        return np.round(restored / self.q) * self.q

    def encoded_bounds(self):
        return None, None

    def serialize_as_nni(self):
        return {
            '_type': 'qlognormal',
            '_value': [self.mean, self.std, self.q]
        }


class ParamGenerator:
    def __init__(self, capsule):
        self.capsule = capsule

    def const(self, key, default, value=None):
        if value is None:
            value = default
        return self._make_param_wrapper(key, ConstParam(self.capsule, key, default, value))

    def choice(self, key, default, choices=None, prob=None):
        if choices is None:
            choices = default
            default = choices[0]
        return self._make_param_wrapper(key, ChoiceParam(self.capsule, key, default, choices, prob))

    def ordered(self, key, default, choices=None, prob=None):
        if choices is None:
            choices = default
            default = choices[0]
        return self._make_param_wrapper(key, OrderedParam(self.capsule, key, default, choices, prob))

    def randint(self, key, default=None, *, low, high):
        if default is None:
            default = low
        return self._make_param_wrapper(key, RandomIntParam(self.capsule, key, default, low, high))

    def unif(self, key, default=None, *, low, high):
        if default is None:
            default = low
        return self._make_param_wrapper(key, UniformParam(self.capsule, key, default, low, high))

    def qunif(self, key, default=None, *, low, high, q):
        if default is None:
            default = low
        return self._make_param_wrapper(key, QUniformParam(self.capsule, key, default, low, high, q))

    def logunif(self, key, default=None, *, low, high):
        if default is None:
            default = low
        return self._make_param_wrapper(key, LogUniformParam(self.capsule, key, default, low, high))

    def qlogunif(self, key, default=None, *, low, high, q):
        if default is None:
            default = low
        return self._make_param_wrapper(key, QLogUniformParam(self.capsule, key, default, low, high, q))

    def normal(self, key, default=None, *, mean, std):
        if default is None:
            default = mean
        return self._make_param_wrapper(key, NormalParam(self.capsule, key, default, mean, std))

    def qnormal(self, key, default=None, *, mean, std, q):
        if default is None:
            default = mean
        return self._make_param_wrapper(key, QNormalParam(self.capsule, key, default, mean, std, q))

    def lognormal(self, key, default=None, *, mean, std):
        if default is None:
            default = np.exp(mean)
        return self._make_param_wrapper(key, LogNormalParam(self.capsule, key, default, mean, std))

    def qlognormal(self, key, default=None, *, mean, std, q):
        if default is None:
            default = np.exp(mean)
        return self._make_param_wrapper(key, QLogNormalParam(self.capsule, key, default, mean, std, q))

    def _make_param_wrapper(self, key, param_wrapper):
        self.capsule._guard_params()
        if key in self.capsule._param_wrappers:
            raise ValueError('duplicate parameter key', key)
        self.capsule._param_wrappers[key] = param_wrapper
        return param_wrapper


if __name__ == '__main__':
    rv = ChoiceParam(None, 'c', 'a', ['a', 'b', 'c'])
    print(rv.encode_as_numerical('a'))
    print(rv.decode_from_numerical(2.1))
    rv = OrderedParam(None, 'c', 'a', ['a', 'b', 'c'])
    print(rv.encode_as_numerical('a'))
    print(rv.decode_from_numerical(1.9))
    rv = UniformParam(None, 'c', 0, -10, 10)
    print(rv.encode_as_numerical(2.1))
    print(rv.decode_from_numerical(0.605))
