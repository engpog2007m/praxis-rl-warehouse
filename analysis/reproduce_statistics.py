#!/usr/bin/env python3
"""Recompute every statistic reported in the praxis from the result CSVs.

Each figure is recomputed from raw episode records using the same test,
alternative hypothesis and random seed as the original analysis, then
compared with the value reported in Chapter 4.

  OK    recomputed value matches the reported value within rounding
  DIFF  recomputed value differs (investigate before relying on it)
  SKIP  the source CSV is not present

Statistical note: "Wilcoxon rank-sum" in the praxis and the Mann-Whitney U
test implemented by scipy.stats.mannwhitneyu are the same test. Effect size
is the rank-biserial correlation r = 1 - 2U/(n1*n2), as in the original code.

Exit status is 1 if any comparison is DIFF, otherwise 0.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.environ.get('PRAXIS_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data')


class Report:
    def __init__(self):
        self.ok = self.diff = self.skip = 0

    def section(self, title):
        print(f'\n{"=" * 76}\n{title}\n{"=" * 76}')

    def skip_file(self, name):
        self.skip += 1
        print(f'  SKIP  {name} not found in data/')

    def check(self, label, got, reported, tol=None, below=None):
        if below is not None:
            good = got < below
            shown = f'< {below:g}'
        else:
            # +1e-9 absorbs floating-point noise when the reported value was
            # rounded and the true value sits exactly at the rounding boundary
            good = abs(got - reported) <= tol + 1e-9
            shown = f'{reported:g}'
        self.ok += good
        self.diff += not good
        print(f"  {'OK  ' if good else 'DIFF'}  {label:<48} "
              f"got={got:<12.6g} reported={shown}")

    def note(self, text):
        print(f'        {text}')


def load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def as_float(s):
    if s.dtype == object:
        s = s.map({'True': 1.0, 'False': 0.0, True: 1.0, False: 0.0})
    return s.astype(float)


def mwu(a, b, alternative):
    W, p = stats.mannwhitneyu(a, b, alternative=alternative)
    return p, 1 - (2 * W) / (len(a) * len(b))


# ------------------------------------------------------------------ Era 2
def era2_four_way(R):
    R.section('Era 2 — four-way evaluation, LiDAR + peer obstacle '
              '(four_way_lidar_results.csv)')
    df = load('four_way_lidar_results.csv')
    if df is None:
        return R.skip_file('four_way_lidar_results.csv')

    reported = {
        'PPO_guided':   (0.2950, 0.1825, 0.5225, -98.3088),
        'PPO_unguided': (0.5625, 0.3738, 0.0638, -90.9731),
        'BC_ml':        (0.4938, 0.3988, 0.1075, -104.3025),
        'Nav2_DWB':     (0.4538, 0.4112, 0.1350, -108.9568),
    }
    for s, (comp, coll, tout, rew) in reported.items():
        sub = df[df['system'] == s]
        R.check(f'{s} completion rate', as_float(sub['completed']).mean(), comp, 5e-5)
        R.check(f'{s} collision rate', as_float(sub['collided']).mean(), coll, 5e-5)
        R.check(f'{s} timeout rate', as_float(sub['timed_out']).mean(), tout, 5e-5)
        R.check(f'{s} mean reward', sub['reward'].mean(), rew, 5e-5)

    print('\n  Pairwise vs Nav2_DWB (two-sided)')
    dwb = df[df['system'] == 'Nav2_DWB']
    vs = {'PPO_guided': (None, None), 'PPO_unguided': (None, 0.1247),
          'BC_ml': (0.1092, 0.6107)}
    for s, (pc_rep, pcol_rep) in vs.items():
        sub = df[df['system'] == s]
        pc, _ = mwu(as_float(sub['completed']), as_float(dwb['completed']), 'two-sided')
        pcol, _ = mwu(as_float(sub['collided']), as_float(dwb['collided']), 'two-sided')
        if pc_rep is None:
            R.check(f'{s} completion p vs DWB', pc, None, below=5e-5)
        else:
            R.check(f'{s} completion p vs DWB', pc, pc_rep, 5e-5)
        if pcol_rep is None:
            R.check(f'{s} collision p vs DWB', pcol, None, below=5e-5)
        else:
            R.check(f'{s} collision p vs DWB', pcol, pcol_rep, 5e-5)

    print('\n  H3 — guided vs unguided')
    g = df[df['system'] == 'PPO_guided']
    u = df[df['system'] == 'PPO_unguided']
    pc, _ = mwu(as_float(g['completed']), as_float(u['completed']), 'two-sided')
    pcol, rcol = mwu(as_float(g['collided']), as_float(u['collided']), 'less')
    R.check('H3 completion p (two-sided)', pc, None, below=1e-4)
    R.check('H3 collision p (one-sided, guided < unguided)', pcol, None, below=1e-4)
    R.check('H3 collision effect size r', rcol, 0.191, 5e-4)
    R.note('Aggregate only. See the goal-direction checks below before reading this as a safety gain.')


def era2_core_and_direction(R):
    R.section('Era 2 — core experiment vs behavioural cloning, and H3 by goal direction '
              '(four_way_lidar_results.csv)')
    df = load('four_way_lidar_results.csv')
    if df is None:
        return R.skip_file('four_way_lidar_results.csv')
    need = {'cycle_time', 'episode'}
    if not need.issubset(df.columns):
        R.skip += 1
        print(f'  SKIP  columns {sorted(need - set(df.columns))} not in four_way_lidar_results.csv')
        return

    # Same procedure and random stream as analysis/bc_ci.py (seed 42, 2,000 resamples,
    # completion first, throughput second), so the intervals match exactly.
    df = df.copy()
    df['completed'] = as_float(df['completed'])
    rng = np.random.default_rng(42)
    B = 2000
    U = df[df.system == 'PPO_unguided'].reset_index(drop=True)
    C = df[df.system == 'BC_ml'].reset_index(drop=True)

    def rel(a, b, f):
        pt = 100 * (f(a) - f(b)) / f(b)
        bs = []
        for _ in range(B):
            x = a.iloc[rng.integers(0, len(a), len(a))]
            y = b.iloc[rng.integers(0, len(b), len(b))]
            bs.append(100 * (f(x) - f(y)) / f(y))
        return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    comp = lambda g: g.completed.mean()
    tput = lambda g: 1000 * g.completed.sum() / g.cycle_time.sum()
    print('\n  Unguided PPO vs behavioural cloning (bootstrap 95% CI)')
    for name, f, (pt_r, lo_r, hi_r) in (('H1 completion', comp, (13.9, 4.5, 24.7)),
                                        ('H2 throughput', tput, (24.3, 15.5, 34.3))):
        pt, lo, hi = rel(U, C, f)
        R.check(f'{name} vs BC, relative (%)', pt, pt_r, 0.05)
        R.check(f'{name} vs BC, CI lower (%)', lo, lo_r, 0.05)
        R.check(f'{name} vs BC, CI upper (%)', hi, hi_r, 0.05)

    print('\n  H3 — guided vs unguided in the direction the guided policy attempted')
    df['direction'] = df['episode'] % 2
    g_all = df[df.system == 'PPO_guided']
    attempted = max((0, 1), key=lambda d: g_all[g_all.direction == d]['completed'].mean())
    g = g_all[g_all.direction == attempted]
    u = df[(df.system == 'PPO_unguided') & (df.direction == attempted)]
    other = g_all[g_all.direction != attempted]
    R.check('H3 guided timeout rate, other direction', as_float(other['timed_out']).mean(), 1.0, 5e-4)
    R.check('H3 attempted dir: guided completion', g['completed'].mean(), 0.590, 5e-4)
    R.check('H3 attempted dir: unguided completion', u['completed'].mean(), 0.588, 5e-4)
    R.check('H3 attempted dir: guided collision', as_float(g['collided']).mean(), 0.365, 5e-4)
    R.check('H3 attempted dir: unguided collision', as_float(u['collided']).mean(), 0.355, 5e-4)
    pc, _ = mwu(g['completed'], u['completed'], 'two-sided')
    pcol, rcol = mwu(as_float(g['collided']), as_float(u['collided']), 'two-sided')
    R.check('H3 attempted dir: completion p', pc, 0.94, 5e-3)
    R.check('H3 attempted dir: collision p', pcol, 0.77, 5e-3)
    R.check('H3 attempted dir: collision effect size r', rcol, -0.010, 5e-4)
    R.note('Verdict: H3 partially supported. The aggregate collision reduction comes from '
           'the guided policy timing out in one direction, not from safer driving.')


# ------------------------------------------------------------------ Era 1
def era1_physics_and_h1(R):
    R.section('Era 1 — physics layer PPO vs Nav2 DWB and H1 '
              '(gazebo_trained_results_v4.csv)')
    df = load('gazebo_trained_results_v4.csv')
    if df is None:
        return R.skip_file('gazebo_trained_results_v4.csv')
    ppo = df[df['system'] == 'PPO_gazebo']
    dwb = df[df['system'] == 'Nav2_DWB']

    for metric, med_p, med_d, p_rep in [('reward', 77.96, 58.70, 0.0035),
                                         ('efficiency', 0.886, 0.645, 0.0078),
                                         ('cycle_time', 102.0, 111.5, 0.0160)]:
        if metric not in df.columns:
            continue
        R.check(f'PPO median {metric}', ppo[metric].median(), med_p,
                0.006 if metric != 'efficiency' else 6e-4)
        R.check(f'DWB median {metric}', dwb[metric].median(), med_d,
                0.006 if metric != 'efficiency' else 6e-4)
        p, _ = mwu(ppo[metric], dwb[metric], 'two-sided')
        R.check(f'{metric} p (two-sided)', p, p_rep, 6e-5)

    R.check('PPO overall completion', as_float(ppo['completed']).mean(), 0.586, 6e-4)
    R.check('DWB overall completion', as_float(dwb['completed']).mean(), 0.526, 6e-4)

    print('\n  H1 — completion by sensor noise')
    for sigma, cp, cd in [(0.00, 0.572, 0.524), (0.05, 0.608, 0.560),
                          (0.15, 0.580, 0.516), (0.30, 0.584, 0.504)]:
        sp = as_float(ppo[np.isclose(ppo['noise_sigma'], sigma)]['completed']).mean()
        sd = as_float(dwb[np.isclose(dwb['noise_sigma'], sigma)]['completed']).mean()
        R.check(f'sigma={sigma:.2f} PPO completion', sp, cp, 6e-4)
        R.check(f'sigma={sigma:.2f} DWB completion', sd, cd, 6e-4)

    hi = df[df['noise_sigma'] >= 0.15]
    p = as_float(hi[hi['system'] == 'PPO_gazebo']['completed'])
    d = as_float(hi[hi['system'] == 'Nav2_DWB']['completed'])
    adv = 100 * (p.mean() - d.mean()) / d.mean()
    pv, r = mwu(p, d, 'greater')
    R.check('H1 PPO completion, sigma >= 0.15', p.mean(), 0.5820, 5e-5)
    R.check('H1 DWB completion, sigma >= 0.15', d.mean(), 0.5100, 5e-5)
    R.check('H1 relative advantage (%)', adv, 14.12, 5e-3)
    R.check('H1 p (one-sided, PPO > DWB)', pv, 0.011149, 5e-7)
    R.check('H1 effect size r', r, -0.0720, 5e-5)

    # Bootstrap reproduced with the original seed and draw order: all 2000
    # DWB resample means are drawn first, then one PPO resample per mean.
    rng = np.random.default_rng(42)
    boot = [100 * (rng.choice(p, len(p), True).mean() - m) / m
            for m in [rng.choice(d, len(d), True).mean() for _ in range(2000)]]
    R.check('H1 95% CI lower bound (%)', np.percentile(boot, 2.5), 2.3, 0.05)
    R.check('H1 95% CI upper bound (%)', np.percentile(boot, 97.5), 27.8, 0.05)
    R.note('Point estimate clears the 10% threshold; the CI lower bound does not.')


def era1_four_way(R):
    R.section('Era 1 — four-way evaluation, no obstacle (four_way_results.csv)')
    df = load('four_way_results.csv')
    if df is None:
        return R.skip_file('four_way_results.csv')
    reported = {'PPO_guided': (86.8237, 0.8418, 0.925),
                'PPO_unguided': (85.2462, 0.8218, 0.923),
                'BC_ml': (76.6341, 0.6635, 0.848),
                'Nav2_DWB': (86.2487, 0.7702, 0.890)}
    for s, (rew, eff, comp) in reported.items():
        sub = df[df['system'] == s]
        R.check(f'{s} mean reward', sub['reward'].mean(), rew, 6e-5)
        R.check(f'{s} mean efficiency', sub['efficiency'].mean(), eff, 6e-5)
        R.check(f'{s} completion', as_float(sub['completed']).mean(), comp, 6e-4)

    dwb = df[df['system'] == 'Nav2_DWB']['reward']
    for s, p_rep in [('PPO_guided', 0.598637), ('PPO_unguided', 0.852861),
                     ('BC_ml', 0.001909)]:
        p, _ = mwu(df[df['system'] == s]['reward'], dwb, 'two-sided')
        R.check(f'{s} reward p vs DWB', p, p_rep, 6e-7)

    g = df[df['system'] == 'PPO_guided']
    u = df[df['system'] == 'PPO_unguided']
    p, r = mwu(g['reward'], u['reward'], 'greater')
    R.check('H3 (no obstacle) reward p', p, 0.3701, 6e-5)
    R.check('H3 (no obstacle) reward r', r, -0.0086, 6e-5)
    p, _ = mwu(as_float(g['completed']), as_float(u['completed']), 'greater')
    R.check('H3 (no obstacle) completion p', p, 0.4330, 6e-5)
    for sigma, p_rep in [(0.0, 0.2545), (0.05, 0.6695), (0.15, 0.3742), (0.30, 0.4607)]:
        gs = g[np.isclose(g['noise_sigma'], sigma)]['reward']
        us = u[np.isclose(u['noise_sigma'], sigma)]['reward']
        p, _ = mwu(gs, us, 'greater')
        R.check(f'H3 (no obstacle) sigma={sigma:.2f} p', p, p_rep, 6e-5)


def era1_h3_appendix(R):
    R.section('Era 1 — H3 null tests, appendix')
    specs = [
        ('h3_test_a_safety.csv', 'violation_rate', 'guided', 'unguided', 'less', 0.000108, 5e-7),
        ('h3_test_b_ood.csv', 'completed', 'guided', 'unguided', 'greater', 0.949878, 5e-7),
        ('h3_roadmap_results.csv', 'completed', 'roadmap_followed', 'roadmap_skipped',
         'greater', 0.562416, 5e-7),
    ]
    for fname, col, a, b, alt, p_rep, tol in specs:
        df = load(fname)
        if df is None:
            R.skip_file(fname)
            continue
        p, _ = mwu(as_float(df[df['system'] == a][col]),
                   as_float(df[df['system'] == b][col]), alt)
        R.check(f'{fname} p', p, p_rep, tol)

    df = load('h3_stress_tiers.csv')
    if df is None:
        return R.skip_file('h3_stress_tiers.csv')
    for tier, p_rep in [('low_load', 0.9538), ('medium', 0.4284),
                        ('high', 0.5461), ('congested', 0.3822)]:
        t = df[df['tier'] == tier]
        p, _ = mwu(as_float(t[t['system'] == 'PPO_guided']['completed']),
                   as_float(t[t['system'] == 'PPO_unguided']['completed']), 'greater')
        R.check(f'stress tier {tier} p', p, p_rep, 6e-5)


def main():
    print(f'Data directory: {DATA}')
    R = Report()
    era2_four_way(R)
    era2_core_and_direction(R)
    era1_physics_and_h1(R)
    era1_four_way(R)
    era1_h3_appendix(R)
    print(f'\n{"=" * 76}')
    print(f'SUMMARY  OK={R.ok}  DIFF={R.diff}  SKIP={R.skip}')
    if R.diff:
        print('At least one recomputed value differs from the reported value.')
    return 1 if R.diff else 0


if __name__ == '__main__':
    sys.exit(main())
