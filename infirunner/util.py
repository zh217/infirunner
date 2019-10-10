import random
import string
import datetime
import json
import box
import sys


def make_trial_id():
    rd_id = ''.join(random.choice(string.ascii_letters) for _ in range(6))
    trial_id = f'{datetime.datetime.now():%y%m%d_%H%M%S}_{rd_id}'
    return trial_id


class UniformBernoulli:
    def __init__(self, p):
        self.p = p
        self.n_true = 0
        self.n_false = 0

    def __call__(self):
        if self.n_true == self.n_false == 0:
            ret = self.p > 0.5
        else:
            ret = self.n_true / (self.n_true + self.n_false) < self.p
        if ret:
            self.n_true += 1
        else:
            self.n_false += 1
        return ret


def load_state(file):
    with open(file) as f:
        orig = json.load(f)['params']
    ret = box.Box()
    for k, v in orig.items():
        ks = k.split('.')
        parent = ret
        for k_seg in ks[:-1]:
            parent = parent.setdefault(k_seg, {})
        parent[ks[-1]] = v
    return ret


def stride_list(orig, world_size, rank):
    l = len(orig)
    orig = orig[:l // world_size * world_size]
    return orig[rank::world_size]


def log(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


if __name__ == '__main__':
    print(stride_list(list(range(100)), 10, 3))
    print(stride_list(list(range(99)), 10, 3))
