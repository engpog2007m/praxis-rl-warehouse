"""
Extract empirical per-task travel-time and efficiency distributions
from the v4 Gazebo evaluation, for use as edge-traversal costs in the
fleet-level discrete-event simulation.
"""
import os, json
import numpy as np
import pandas as pd

SRC = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data/gazebo_trained_results_v4.csv')
OUT = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data/fleet_distributions.json')

GOAL_DISTANCE_M = {1: 3.0, 2: 3.5, 3: 2.5, 4: 1.2, 1001: 1.5}
SECONDS_PER_TICK = 0.02

df = pd.read_csv(SRC)
print(f"Loaded {len(df):,} episode records")
print(f"Systems: {sorted(df['system'].unique())}")
print(f"Tasks  : {sorted(df['task_id'].unique())}")
print()

dists = {}
for system in sorted(df['system'].unique()):
    dists[system] = {}
    sdf = df[df['system'] == system]
    for tid in sorted(df['task_id'].unique()):
        tdf = sdf[sdf['task_id'] == tid]
        done = tdf[tdf['completed'] == True]
        if len(done) == 0:
            done = tdf
        ticks = done['cycle_time'].values.astype(float)
        secs  = ticks * SECONDS_PER_TICK
        dist_m = GOAL_DISTANCE_M[tid]
        speeds = np.where(secs > 0, dist_m / secs, 0.0)
        dists[system][str(tid)] = {
            'n_episodes'     : int(len(tdf)),
            'n_completed'    : int(tdf['completed'].sum()),
            'completion_rate': round(float(tdf['completed'].mean()), 4),
            'goal_distance_m': dist_m,
            'ticks_mean'     : round(float(np.mean(ticks)), 2),
            'ticks_std'      : round(float(np.std(ticks)), 2),
            'ticks_p25'      : round(float(np.percentile(ticks, 25)), 2),
            'ticks_p50'      : round(float(np.percentile(ticks, 50)), 2),
            'ticks_p75'      : round(float(np.percentile(ticks, 75)), 2),
            'seconds_mean'   : round(float(np.mean(secs)), 4),
            'seconds_std'    : round(float(np.std(secs)), 4),
            'speed_mean_mps' : round(float(np.mean(speeds)), 4),
            'speed_std_mps'  : round(float(np.std(speeds)), 4),
            'efficiency_mean': round(float(tdf['efficiency'].mean()), 4),
            'efficiency_std' : round(float(tdf['efficiency'].std()), 4),
            'reward_mean'    : round(float(tdf['reward'].mean()), 4),
            'tick_samples'   : [float(x) for x in ticks],
        }

for system in dists:
    all_speeds = []
    for tid, d in dists[system].items():
        all_speeds.extend([d['speed_mean_mps']] * max(d['n_completed'], 1))
    dists[system]['_aggregate'] = {
        'speed_mean_mps': round(float(np.mean(all_speeds)), 4),
        'speed_std_mps' : round(float(np.std(all_speeds)), 4),
    }

meta = {
    'source_file'     : 'gazebo_trained_results_v4.csv',
    'seconds_per_tick': SECONDS_PER_TICK,
    'goal_distance_m' : GOAL_DISTANCE_M,
    'note': ('Travel times from single-robot Gazebo Harmonic physics '
             'evaluation. Effective speed = goal_distance/(ticks*dt). '
             'Used as free-flow edge traversal speed in the fleet-level '
             'simulation; congestion delay applied separately.'),
}

with open(OUT, 'w') as f:
    json.dump({'metadata': meta, 'distributions': dists}, f, indent=2)

print("=" * 66)
print("EMPIRICAL DISTRIBUTIONS BY SYSTEM AND TASK")
print("=" * 66)
print(f"{'system':<12}{'task':>6}{'compl%':>9}{'ticks':>9}{'sec':>9}{'m/s':>9}")
print("-" * 66)
for system in dists:
    for tid, d in dists[system].items():
        if tid == '_aggregate': continue
        print(f"{system:<12}{tid:>6}{d['completion_rate']*100:>8.1f}%"
              f"{d['ticks_mean']:>9.1f}{d['seconds_mean']:>9.3f}"
              f"{d['speed_mean_mps']:>9.3f}")
    print("-" * 66)
print()
print("AGGREGATE FREE-FLOW SPEED")
for system in dists:
    agg = dists[system]['_aggregate']
    print(f"  {system:<12} {agg['speed_mean_mps']:.4f} +/- {agg['speed_std_mps']:.4f} m/s")
print()
print(f"Saved -> {OUT}")
