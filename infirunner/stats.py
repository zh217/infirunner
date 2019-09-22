import pandas as pd
import os
import json
import math


def get_stats(exp_dir, collect_budget=1, ignore_nan=True):
    trials = [d for d in os.listdir(exp_dir) if len(d) == 20]
    collected = []

    for trial in trials:
        try:
            with open(os.path.join(exp_dir, trial, 'metric.tsv'), 'rb') as f:
                for line in f:
                    budget, _, metric = (line.split())
                    budget = int(budget)
                    if budget == collect_budget:
                        metric = float(metric)
                        if ignore_nan and not math.isfinite(metric):
                            break
                        with open(os.path.join(exp_dir, trial, 'saves', f'{budget:05d}', 'state.json'), 'rb') as jf:
                            params = json.load(jf)['params']
                        params['TRIAL'] = trial
                        params['METRIC'] = metric
                        collected.append(params)
                        break
        except:
            continue

    return pd.DataFrame(collected).set_index('TRIAL')
