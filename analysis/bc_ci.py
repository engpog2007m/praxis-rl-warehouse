import numpy as np, pandas as pd
d = pd.read_csv('data/four_way_lidar_results.csv')
d['completed'] = d['completed'].astype(float)
rng = np.random.default_rng(42); B = 2000
U = d[d.system == 'PPO_unguided'].reset_index(drop=True)
C = d[d.system == 'BC_ml'].reset_index(drop=True)
def rel(a, b, f):
    pt = 100 * (f(a) - f(b)) / f(b); bs = []
    for _ in range(B):
        x = a.iloc[rng.integers(0, len(a), len(a))]
        y = b.iloc[rng.integers(0, len(b), len(b))]
        bs.append(100 * (f(x) - f(y)) / f(y))
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
comp = lambda g: g.completed.mean()
tput = lambda g: 1000 * g.completed.sum() / g.cycle_time.sum()
print('H1 completion, unguided vs BC: %+.1f%% [%.1f, %.1f]' % rel(U, C, comp))
print('H2 throughput, unguided vs BC: %+.1f%% [%.1f, %.1f]' % rel(U, C, tput))
