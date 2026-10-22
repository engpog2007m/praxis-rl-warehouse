"""Core-experiment figures 4.5-4.8: unguided PPO vs behavioral cloning vs DWB reference controller.
Fig 4.5 uses the unguided training log; Figs 4.6-4.8 use the outcome counts from the
final-stage evaluation (n = 800 per system; 200 per noise level)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi':300,'savefig.dpi':300,'font.size':11,'font.family':'DejaVu Sans',
  'axes.grid':True,'grid.alpha':0.25,'grid.linestyle':'-','axes.spines.top':False,'axes.spines.right':False,
  'axes.edgecolor':'#9CA3AF','xtick.color':'#374151','ytick.color':'#374151','axes.labelcolor':'#111827'})
ROOT = os.environ.get('PRAXIS_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'results', 'figures') + os.sep
os.makedirs(OUT, exist_ok=True)
TRAIN = os.path.join(ROOT, 'data', 'lidar_unguided_training_curves.csv')
SYS = ['U', 'BC', 'DWB']
LBL = {'U':'PPO (unguided)', 'BC':'Behavioral cloning', 'DWB':'DWB reference controller'}
COL = {'U':'#2E5EAA', 'BC':'#C4622D', 'DWB':'#6B7280'}
MK  = {'U':'s', 'BC':'^', 'DWB':'D'}
def wilson(k, n, z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*(c-h), 100*(c+h)
def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT+name, bbox_inches='tight', facecolor='white'); plt.close(fig); print('wrote', name)

# ---- Fig 4.5 training (unguided only) ----
t = pd.read_csv(TRAIN)
first = t.index[t.collision_rate > 0][0]          # readings before the first terminal outcome are not measurements
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
a1.plot(t.timestep[first:], 100*t.collision_rate[first:], color=COL['U'], lw=1.6)
a2.plot(t.timestep, t.mean_length, color=COL['U'], lw=1.6)
for a in (a1, a2):
    for b in range(100000, 500001, 100000): a.axvline(b, color='#111827', lw=0.6, ls=':')
    a.set_xlabel('Training steps'); a.set_xlim(0, 500000)
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
a1.set_ylabel('Collision rate, last 40 terminal outcomes (%)'); a1.set_ylim(0, 100)
a2.set_ylabel('Mean episode length (steps)'); a2.set_ylim(0, 210)
a1.set_title('(a) Collision rate', fontsize=11, loc='left'); a2.set_title('(b) Episode length', fontsize=11, loc='left')
save(fig, 'fig4_5_training_unguided.png')

# ---- outcome counts (n = 800 each) ----
n = 800
K = {'U':  {'completed':450, 'collided':299, 'timed_out':51},
     'BC': {'completed':395, 'collided':319, 'timed_out':86},
     'DWB':{'completed':363, 'collided':329, 'timed_out':108}}
for s in K: assert sum(K[s].values()) == n

# ---- Fig 4.6 outcomes ----
OC = [('completed','Completed','#2C7A4B'), ('collided','Collided','#B03030'), ('timed_out','Timed out','#8C8C8C')]
fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(3); w = 0.26
for j, (key, name, color) in enumerate(OC):
    v = [100*K[s][key]/n for s in SYS]; ci = [wilson(K[s][key], n) for s in SYS]
    lo = [vi-c[0] for vi, c in zip(v, ci)]; hi = [c[1]-vi for vi, c in zip(v, ci)]
    ax.bar(x+(j-1)*w, v, w-0.02, color=color, label=name, yerr=[lo, hi], capsize=3, error_kw={'lw':0.8,'ecolor':'#111827'})
    for xi, vi, h in zip(x+(j-1)*w, v, hi): ax.text(xi, vi+h+1, f'{vi:.1f}', ha='center', fontsize=8.5, color='#111827')
ax.set_xticks(x); ax.set_xticklabels([LBL[s] for s in SYS]); ax.set_ylabel('Episodes (%)'); ax.set_ylim(0, 68)
ax.legend(loc='upper right', ncol=3, frameon=False); ax.grid(axis='x', visible=False)
save(fig, 'fig4_6_outcomes_core.png')

# ---- Fig 4.7 completion by noise (n = 200 per point) ----
sig = [0.00, 0.05, 0.15, 0.30]
PCT = {'U':[56.5, 62.0, 53.5, 53.0], 'BC':[51.5, 49.0, 53.5, 43.5], 'DWB':[47.5, 53.5, 42.0, 38.5]}
for s in PCT: assert abs(np.mean(PCT[s]) - 100*K[s]['completed']/n) < 0.01
fig, ax = plt.subplots(figsize=(9, 5))
off = {'U':-0.007, 'BC':0.0, 'DWB':0.007}
for s in SYS:
    k = [round(p*2) for p in PCT[s]]; ci = [wilson(kk, 200) for kk in k]
    ax.errorbar([g+off[s] for g in sig], PCT[s], yerr=[[p-c[0] for p, c in zip(PCT[s], ci)], [c[1]-p for p, c in zip(PCT[s], ci)]],
                marker=MK[s], color=COL[s], lw=2, ms=8, capsize=4, label=LBL[s], mec='white', mew=1)
ax.set_xticks(sig); ax.set_xlabel('Sensor noise σ'); ax.set_ylabel('Task completion (%)'); ax.set_ylim(25, 75)
ax.legend(ncol=3, loc='upper center', frameon=False)
save(fig, 'fig4_7_completion_noise_core.png')

# ---- Fig 4.8 throughput ----
T = {'U':4.315, 'BC':3.471, 'DWB':3.226}
REL = {'U':(33.7, 23.4, 45.8), 'BC':(7.6, -1.6, 18.7)}   # vs DWB, bootstrap 95% CI (2,000 resamples, seed 42)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
a1.bar(range(3), [T[s] for s in SYS], 0.6, color=[COL[s] for s in SYS])
for i, s in enumerate(SYS): a1.text(i, T[s]+0.07, f'{T[s]:.2f}', ha='center', fontsize=9.5, color='#111827')
a1.set_xticks(range(3)); a1.set_xticklabels([LBL[s].replace(' reference', '\nreference') for s in SYS], fontsize=9.5)
a1.set_ylabel('Completed tasks per 1,000 control steps'); a1.set_ylim(0, 5); a1.grid(axis='x', visible=False)
a1.set_title('(a) Throughput', fontsize=11, loc='left')
rs = ['U', 'BC']; v = [REL[s][0] for s in rs]
a2.bar(range(2), v, 0.5, color=[COL[s] for s in rs],
       yerr=[[REL[s][0]-REL[s][1] for s in rs], [REL[s][2]-REL[s][0] for s in rs]], capsize=5, error_kw={'lw':0.9,'ecolor':'#111827'})
a2.axhline(0, color='#111827', lw=1)
for i, s in enumerate(rs): a2.text(i, REL[s][2]+2, f'{REL[s][0]:+.1f}%', ha='center', fontsize=9.5, color='#111827')
a2.set_xticks(range(2)); a2.set_xticklabels([LBL[s] for s in rs], fontsize=9.5)
a2.set_ylabel('Change vs DWB reference controller (%), 95% CI'); a2.set_ylim(-10, 55); a2.grid(axis='x', visible=False)
a2.set_title('(b) Relative to DWB reference controller', fontsize=11, loc='left')
save(fig, 'fig4_8_throughput_core.png')
