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


def plot_stats(folder, budget=1, replace_with_log=('opt.lr',), metric_column='METRIC',
               n_cols=2, fig_height=6, fig_width=15, metric_bound=None, continuous_threshold=6):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set()

    stats = get_stats(folder, budget)
    for k in replace_with_log:
        stats['log10_' + k] = np.log10(stats[k])
    single_vals = list(replace_with_log)
    for c in stats.columns:
        if stats[c].nunique() == 1:
            single_vals.append(c)
            print(f'{c:30} {stats[c].iloc[0]}')
    stats = stats.drop(columns=single_vals).sort_values(metric_column)
    if metric_bound is not None:
        stats = stats[stats[metric_column] < metric_bound]

    n_rows = len(stats.columns) // n_cols
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    fig.set_figheight(fig_height * n_rows)
    fig.set_figwidth(fig_width)

    i = 0
    for c in stats.columns:
        if c != metric_column:
            if stats[c].dtype != 'float64' or stats[c].nunique() < continuous_threshold:
                sns.swarmplot(x=c, y=metric_column, data=stats, ax=axs[i // 2][i % 2])
            else:
                sns.scatterplot(x=c, y=metric_column, data=stats, ax=axs[i // 2][i % 2])
            i += 1
    return fig
