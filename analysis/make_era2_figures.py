#!/usr/bin/env python3
"""Generate Chapter 4 figures 4.5 to 4.9 from the Era 2 data files.

Inputs (data/):
  four_way_lidar_results.csv
  lidar_guided_training_curves.csv, lidar_unguided_training_curves.csv
Outputs (results/figures/):
  fig4_5_training.png          collision rate and episode length over training
  fig4_6_outcomes.png          outcome rates with Wilson 95% CIs
  fig4_7_completion_noise.png  completion by sensor noise with Wilson 95% CIs
  fig4_8_throughput.png        throughput, absolute and relative to Nav2 DWB
  fig4_9_direction.png         outcomes by goal direction

Figures carry no embedded titles or numbers; use Word captions.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.environ.get('PRAXIS_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'results', 'figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 11, 'axes.grid': True,
                     'grid.alpha': 0.3, 'grid.linestyle': '--', 'axes.spines.top': False,
                     'axes.spines.right': False})
SYS = ['PPO_guided', 'PPO_unguided', 'BC_ml', 'Nav2_DWB']
LBL = {'PPO_guided': 'PPO\n(guided)', 'PPO_unguided': 'PPO\n(unguided)',
       'BC_ml': 'Behavioural\ncloning', 'Nav2_DWB': 'Nav2 DWB'}
COL = {'PPO_guided': '#2E5EAA', 'PPO_unguided': '#7BA7D9', 'BC_ml': '#C4622D', 'Nav2_DWB': '#6B7280'}
OC = {'completed': ('Completed', '#2C7A4B'), 'collided': ('Collided', '#B03030'),
      'timed_out': ('Timed out', '#8C8C8C')}
B = 2000


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def fig_training():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for s, f in [('PPO_guided', 'lidar_guided_training_curves.csv'),
                 ('PPO_unguided', 'lidar_unguided_training_curves.csv')]:
        path = os.path.join(DATA, f)
        if not os.path.exists(path):
            print('missing', f)
            continue
        t = pd.read_csv(path)
        a1.plot(t.timestep, 100 * t.collision_rate, color=COL[s], lw=1.6, label=LBL[s].replace('\n', ' '))
        a2.plot(t.timestep, t.mean_length, color=COL[s], lw=1.6, label=LBL[s].replace('\n', ' '))
    for a in (a1, a2):
        for b in range(100000, 500001, 100000):
            a.axvline(b, color='black', lw=0.6, ls=':')
        a.set_xlabel('Training steps')
        a.legend(frameon=False)
    a1.set_ylabel('Collision rate, last 40 terminal outcomes (%)')
    a2.set_ylabel('Mean episode length (steps)')
    a1.set_title('(a) Collision rate', fontsize=11)
    a2.set_title('(b) Episode length', fontsize=11)
    save(fig, 'fig4_5_training.png')


def fig_outcomes(d):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(4)
    w = 0.26
    for j, (c, (name, color)) in enumerate(OC.items()):
        v, lo, hi = [], [], []
        for s in SYS:
            g = d[d.system == s]
            k, n = int(g[c].sum()), len(g)
            a, b = wilson(k, n)
            v.append(100 * k / n)
            lo.append(100 * k / n - a)
            hi.append(b - 100 * k / n)
        ax.bar(x + (j - 1) * w, v, w, color=color, edgecolor='black', lw=0.6, label=name,
               yerr=[lo, hi], capsize=3, error_kw={'lw': 0.8})
        for xi, vi, e in zip(x + (j - 1) * w, v, hi):
            ax.text(xi, vi + e + 1, f'{vi:.1f}', ha='center', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LBL[s] for s in SYS])
    ax.set_ylabel('Episodes (%)')
    ax.set_ylim(0, 65)
    ax.legend(loc='upper right', ncol=3, frameon=False)
    save(fig, 'fig4_6_outcomes.png')


def fig_noise(d):
    mk = {'PPO_guided': 'o', 'PPO_unguided': 's', 'BC_ml': '^', 'Nav2_DWB': 'D'}
    sig = sorted(d.noise_sigma.unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in SYS:
        v, lo, hi = [], [], []
        for x in sig:
            g = d[(d.system == s) & np.isclose(d.noise_sigma, x)]
            k, n = int(g.completed.sum()), len(g)
            a, b = wilson(k, n)
            v.append(100 * k / n)
            lo.append(100 * k / n - a)
            hi.append(b - 100 * k / n)
        ax.errorbar(sig, v, yerr=[lo, hi], marker=mk[s], color=COL[s], lw=2, ms=7, capsize=4,
                    label=LBL[s].replace('\n', ' '))
    ax.set_xticks(sig)
    ax.set_xlabel('Sensor noise σ')
    ax.set_ylabel('Task completion (%)')
    ax.set_ylim(15, 82)
    ax.legend(ncol=4, loc='upper center', frameon=False)
    save(fig, 'fig4_7_completion_noise.png')


def fig_throughput(d):
    def tput(g):
        return 1000 * g.completed.sum() / g.cycle_time.sum()
    dwb = d[d.system == 'Nav2_DWB'].reset_index(drop=True)
    T = {s: tput(d[d.system == s]) for s in SYS}
    rel = {}
    for s in ['PPO_guided', 'PPO_unguided', 'BC_ml']:
        g = d[d.system == s].reset_index(drop=True)
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(B):
            a = g.iloc[rng.integers(0, len(g), len(g))]
            b = dwb.iloc[rng.integers(0, len(dwb), len(dwb))]
            boot.append(100 * (tput(a) - tput(b)) / tput(b))
        rel[s] = (100 * (T[s] - T['Nav2_DWB']) / T['Nav2_DWB'],
                  np.percentile(boot, 2.5), np.percentile(boot, 97.5))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    a1.bar(range(4), [T[s] for s in SYS], color=[COL[s] for s in SYS], edgecolor='black', lw=0.6)
    for i, s in enumerate(SYS):
        a1.text(i, T[s] + 0.08, f'{T[s]:.2f}', ha='center', fontsize=9)
    a1.set_xticks(range(4))
    a1.set_xticklabels([LBL[s] for s in SYS], fontsize=9.5)
    a1.set_ylabel('Completed tasks per 1,000 control steps')
    a1.set_title('(a) Throughput', fontsize=11)
    rs = list(rel)
    v = [rel[s][0] for s in rs]
    err = [[rel[s][0] - rel[s][1] for s in rs], [rel[s][2] - rel[s][0] for s in rs]]
    a2.bar(range(3), v, color=[COL[s] for s in rs], edgecolor='black', lw=0.6, yerr=err, capsize=5)
    a2.axhline(0, color='black', lw=1)
    for i, s in enumerate(rs):
        a2.text(i, rel[s][2] + 3 if v[i] > 0 else rel[s][1] - 7, f'{v[i]:+.1f}%', ha='center', fontsize=9)
    a2.set_xticks(range(3))
    a2.set_xticklabels([LBL[s] for s in rs], fontsize=9.5)
    a2.set_ylabel('Change vs Nav2 DWB (%), 95% CI')
    a2.set_ylim(-65, 60)
    a2.set_title('(b) Relative to Nav2 DWB', fontsize=11)
    save(fig, 'fig4_8_throughput.png')


def fig_direction(d):
    d = d.assign(direction=d.episode % 2)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    pos, k = [], 0.0
    for s in SYS:
        pos += [k, k + 1]
        k += 2.6
    i = 0
    for s in SYS:
        for dr in (0, 1):
            g = d[(d.system == s) & (d.direction == dr)]
            bottom = 0
            for c, (name, color) in OC.items():
                val = 100 * g[c].mean()
                ax.bar(pos[i], val, 0.8, bottom=bottom, color=color, edgecolor='black', lw=0.5,
                       label=name if i == 0 else None)
                if val >= 6:
                    ax.text(pos[i], bottom + val / 2, f'{val:.1f}', ha='center', va='center',
                            fontsize=8, color='white')
                bottom += val
            i += 1
    ax.set_xticks(pos)
    ax.set_xticklabels(['Direction 0', 'Direction 1'] * 4, fontsize=8.5)
    for n, s in enumerate(SYS):
        ax.text((pos[2 * n] + pos[2 * n + 1]) / 2, -14, LBL[s].replace('\n', ' '), ha='center',
                fontsize=10, fontweight='bold')
    ax.set_ylabel('Episodes (%)')
    ax.set_ylim(0, 108)
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.1), frameon=False)
    fig.subplots_adjust(bottom=0.17)
    save(fig, 'fig4_9_direction.png')


def main():
    d = pd.read_csv(os.path.join(DATA, 'four_way_lidar_results.csv'))
    for c in OC:
        d[c] = d[c].astype(float)
    fig_training()
    fig_outcomes(d)
    fig_noise(d)
    fig_throughput(d)
    fig_direction(d)
    print('figures in', OUT)


if __name__ == '__main__':
    main()
