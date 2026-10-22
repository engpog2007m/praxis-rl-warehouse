#!/usr/bin/env python3
"""Era 2 analysis: every Chapter 4 statistic from four_way_lidar_results.csv.

Tests and definitions are fixed here, before results are read:
  rates       completion, collision, timeout with Wilson 95% CIs
  H1          completion by noise; pooled sigma >= 0.15 vs Nav2 DWB,
              Mann-Whitney one-sided, bootstrap 95% CI (seed 42)
  H2          throughput = completed tasks per 1,000 control steps,
              relative to Nav2 DWB, bootstrap 95% CI (seed 42);
              cycle time of completed episodes, Mann-Whitney two-sided
  H3          guided vs unguided, overall and within the goal direction
              the guided policy attempted; Mann-Whitney two-sided
  RQ1         each system vs Nav2 DWB, Mann-Whitney two-sided
  breakdowns  completion and collision by payload profile and by mass

The RL system for H1 and H2 is unguided PPO, because the guided policy
failed every episode in one goal direction (a separate finding). Guided
results are reported alongside for completeness.

Tables are printed and written to results/tables/era2_*.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.environ.get('PRAXIS_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'results', 'tables')
os.makedirs(OUT, exist_ok=True)
SYSTEMS = ['PPO_guided', 'PPO_unguided', 'BC_ml', 'Nav2_DWB']
RL = 'PPO_unguided'
B = 2000


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def mwu(a, b, alt):
    W, p = stats.mannwhitneyu(a, b, alternative=alt)
    return p, 1 - 2 * W / (len(a) * len(b))


def show(title, df, name):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')
    print(df.to_string())
    df.to_csv(os.path.join(OUT, f'era2_{name}.csv'))


def main():
    d = pd.read_csv(os.path.join(ROOT, 'data', 'four_way_lidar_results.csv'))
    for c in ['completed', 'collided', 'timed_out']:
        d[c] = d[c].astype(float)
    d['direction'] = d['episode'] % 2
    dwb = d[d.system == 'Nav2_DWB']

    # ---------------------------------------------------------------- rates
    rows = []
    for s in SYSTEMS:
        g = d[d.system == s]
        n = len(g)
        r = {'n': n}
        for c in ['completed', 'collided', 'timed_out']:
            k = int(g[c].sum())
            lo, hi = wilson(k, n)
            r[f'{c}_%'] = round(100 * k / n, 1)
            r[f'{c}_95CI'] = f'[{lo:.1f}, {hi:.1f}]'
        rows.append(pd.Series(r, name=s))
    show('Outcome rates with Wilson 95% CIs', pd.DataFrame(rows), 'outcome_rates')

    # ------------------------------------------------------------------ RQ1
    rows = []
    for s in ['PPO_guided', 'PPO_unguided', 'BC_ml']:
        g = d[d.system == s]
        pc, rc = mwu(g.completed, dwb.completed, 'two-sided')
        pk, rk = mwu(g.collided, dwb.collided, 'two-sided')
        rows.append(pd.Series({'completion_diff_pp': round(100 * (g.completed.mean() - dwb.completed.mean()), 1),
                               'completion_p': pc, 'completion_r': round(rc, 3),
                               'collision_diff_pp': round(100 * (g.collided.mean() - dwb.collided.mean()), 1),
                               'collision_p': pk, 'collision_r': round(rk, 3)}, name=f'{s} vs Nav2_DWB'))
    show('RQ1: each system vs Nav2 DWB (Mann-Whitney U, two-sided)', pd.DataFrame(rows), 'rq1_pairwise')

    # ------------------------------------------------------------------- H1
    t = d.groupby(['noise_sigma', 'system']).completed.mean().mul(100).round(1).unstack()[SYSTEMS]
    t[f'{RL} - Nav2_DWB (pp)'] = (t[RL] - t['Nav2_DWB']).round(1)
    show('H1: completion (%) by sensor noise', t, 'h1_by_noise')

    hi = d[d.noise_sigma >= 0.15]
    rows = []
    for s in [RL, 'PPO_guided']:
        p = hi[hi.system == s].completed.values
        q = hi[hi.system == 'Nav2_DWB'].completed.values
        pv, r = mwu(p, q, 'greater')
        rng = np.random.default_rng(42)
        boot = [100 * (rng.choice(p, len(p)).mean() - m) / m
                for m in [rng.choice(q, len(q)).mean() for _ in range(B)]]
        rows.append(pd.Series({'n_each': len(p), 'rl_completion_%': round(100 * p.mean(), 2),
                               'dwb_completion_%': round(100 * q.mean(), 2),
                               'relative_advantage_%': round(100 * (p.mean() - q.mean()) / q.mean(), 2),
                               'boot_95CI_low_%': round(np.percentile(boot, 2.5), 2),
                               'boot_95CI_high_%': round(np.percentile(boot, 97.5), 2),
                               'p_one_sided': pv, 'r': round(r, 4),
                               'meets_10pct_point': 100 * (p.mean() - q.mean()) / q.mean() >= 10,
                               'CI_low_above_10pct': np.percentile(boot, 2.5) >= 10}, name=s))
    show('H1: pooled sigma >= 0.15 vs Nav2 DWB', pd.DataFrame(rows), 'h1_pooled')

    # ------------------------------------------------------------------- H2
    def tput(g):
        return 1000 * g.completed.sum() / g.cycle_time.sum()
    rows = []
    base = tput(dwb)
    for s in SYSTEMS:
        g = d[d.system == s]
        rng = np.random.default_rng(42)
        gi, di = g.reset_index(drop=True), dwb.reset_index(drop=True)
        boot = []
        for _ in range(B):
            a = gi.iloc[rng.integers(0, len(gi), len(gi))]
            b = di.iloc[rng.integers(0, len(di), len(di))]
            boot.append(100 * (tput(a) - tput(b)) / tput(b))
        comp = g[g.completed == 1].cycle_time
        dcomp = dwb[dwb.completed == 1].cycle_time
        p_ct = mwu(comp, dcomp, 'two-sided')[0] if s != 'Nav2_DWB' else np.nan
        rows.append(pd.Series({'tasks_per_1000_steps': round(tput(g), 3),
                               'vs_DWB_%': round(100 * (tput(g) - base) / base, 2),
                               'boot_95CI_low_%': round(np.percentile(boot, 2.5), 2) if s != 'Nav2_DWB' else np.nan,
                               'boot_95CI_high_%': round(np.percentile(boot, 97.5), 2) if s != 'Nav2_DWB' else np.nan,
                               'completed_cycle_time_median': comp.median(),
                               'completed_cycle_time_mean': round(comp.mean(), 1),
                               'cycle_time_p_vs_DWB': p_ct}, name=s))
    show('H2: throughput and cycle time of completed episodes', pd.DataFrame(rows), 'h2_throughput')

    # ------------------------------------------------------------------- H3
    t = d.groupby(['system', 'direction'])[['completed', 'collided', 'timed_out']].mean().mul(100).round(1)
    show('H3: outcomes by goal direction (%)', t, 'h3_by_direction')
    g, u = d[d.system == 'PPO_guided'], d[d.system == 'PPO_unguided']
    att = g.groupby('direction').timed_out.mean().idxmin()
    rows = []
    for label, gg, uu in [('all episodes', g, u),
                          (f'attempted direction ({att}) only', g[g.direction == att], u[u.direction == att])]:
        pc, rc = mwu(gg.completed, uu.completed, 'two-sided')
        pk, rk = mwu(gg.collided, uu.collided, 'two-sided')
        rows.append(pd.Series({'n_each': len(gg),
                               'guided_completed_%': round(100 * gg.completed.mean(), 1),
                               'unguided_completed_%': round(100 * uu.completed.mean(), 1), 'completion_p': pc,
                               'guided_collided_%': round(100 * gg.collided.mean(), 1),
                               'unguided_collided_%': round(100 * uu.collided.mean(), 1), 'collision_p': pk,
                               'collision_r': round(rk, 3)}, name=label))
    show('H3: guided vs unguided (Mann-Whitney U, two-sided)', pd.DataFrame(rows), 'h3_comparison')

    # ----------------------------------------------------------- breakdowns
    for col, name in [('task_id', 'payload'), ('mass_multiplier', 'mass')]:
        t = d.groupby([col, 'system'])[['completed', 'collided']].mean().mul(100).round(1).unstack()
        show(f'Completion and collision (%) by {name}', t, f'by_{name}')

    print(f'\nTables written to {OUT}')


if __name__ == '__main__':
    main()
