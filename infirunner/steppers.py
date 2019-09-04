import time


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
        if self.capsule.is_leader():
            self.capsule.log_scalar(self.key, self.get())
        if flush:
            self.reset()

    def write_to_tb(self, flush=True, start_budget=1):
        if self.capsule.budget_current >= start_budget:
            if self.capsule.is_leader():
                writer = self.capsule.get_tb_writer()
                writer.add_scalar(self.key, self.get(), self.capsule.steps)
        if flush:
            self.reset()

    def write_and_flush(self, start_budget=1):
        self.write_to_log()
        self.write_to_tb(start_budget=start_budget)


class RunningAverageGroup:
    def __init__(self, capsule, keys):
        for k in keys:
            self.__dict__[k] = RunningAverage(capsule, k)


class Timer:
    def __init__(self, *timeouts, unit='seconds'):
        assert timeouts
        assert unit in ('seconds', 'minutes', 'hours')
        if unit == 'minutes':
            timeouts = (t * 60 for t in timeouts)
        if unit == 'hours':
            timeouts = (t * 3600 for t in timeouts)
        self.last_time = time.time()
        self.timeouts = list(timeouts)
        self.timeout = self.get_next_timeout()

    def get_next_timeout(self):
        try:
            return self.timeouts.pop(0)
        except IndexError:
            return self.timeout

    def reset(self):
        self.last_time = time.time()

    def ticked(self):
        now = time.time()
        if now - self.last_time > self.timeout:
            self.last_time = now
            self.timeout = self.get_next_timeout()
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
