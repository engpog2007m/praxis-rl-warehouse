import sys, os
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data')
OUT  = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'results')
os.makedirs(OUT, exist_ok=True)

PPO_C = "#2a78d6"
DWB_C = "#eb6834"
TLABELS = {1:"Standard\nBox", 2:"Lightweight\nPkg", 3:"Heavy\nPallet",
           4:"Fragile\nGlassware", 1001:"Heavy+\nFragile"}
NOISES = [0.00, 0.05, 0.15, 0.30]
MASSES = [1.0, 1.25, 1.50, 2.0]
TASKS  = [1, 2, 3, 4, 1001]
METS   = ['throughput','efficiency','violation_rate','cycle_time']
TITLS  = ['Throughput Rate','Efficiency Index','Safety Violation Rate','Cycle Time (ticks)']

plt.rcParams.update({"font.size":11,"axes.spines.top":False,
    "axes.spines.right":False,"axes.grid":True,
    "grid.alpha":0.3,"grid.linestyle":"--","figure.dpi":300})

df     = pd.read_csv(f'{DATA}/evaluation_results.csv')
curves = pd.read_csv(f'{DATA}/ppo_training_curves.csv')
ppo    = df[df['system']=='PPO']
dwb    = df[df['system']=='Nav2_DWB']

# Fig 4.1 Training convergence
fig, axes = plt.subplots(1, 2, figsize=(12,4.5))
fig.suptitle("Figure 4.1 — PPO Training Convergence (500,000 timesteps)", fontsize=12, fontweight="bold")
ax = axes[0]
ax.plot(curves['timestep'], curves['mean_reward'], color=PPO_C, lw=2, label="Mean reward (20-ep window)")
ax.fill_between(curves['timestep'],
    curves['mean_reward']-curves['std_reward'],
    curves['mean_reward']+curves['std_reward'], alpha=0.2, color=PPO_C)
ax.axhline(0, color='gray', lw=0.8, ls='--', label='Zero reward baseline')
ax.set_xlabel("Simulation timesteps"); ax.set_ylabel("Mean episode reward")
ax.set_title("Reward convergence"); ax.legend(fontsize=9)
ax = axes[1]
ax.plot(curves['timestep'], curves['mean_length'], color='#1baf7a', lw=2)
ax.axhline(90, color='red', lw=0.8, ls='--', label='Throughput threshold (90 ticks)')
ax.set_xlabel("Simulation timesteps"); ax.set_ylabel("Mean episode length (ticks)")
ax.set_title("Episode length over training"); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_1_training_convergence.png', dpi=300, bbox_inches='tight')
plt.close(); print("Fig 4.1 saved")

# Fig 4.2 Per-task comparison
fig, axes = plt.subplots(2, 2, figsize=(14,9))
fig.suptitle("Figure 4.2 — Per-Task Performance: PPO vs Nav2 DWB", fontsize=12, fontweight="bold")
axes = axes.flatten(); x = np.arange(len(TASKS)); w = 0.35
for i, (met, ttl) in enumerate(zip(METS, TITLS)):
    ax = axes[i]
    pv = [np.median(ppo[ppo['task_id']==t][met]) for t in TASKS]
    dv = [np.median(dwb[dwb['task_id']==t][met]) for t in TASKS]
    pe = [float(np.percentile(ppo[ppo['task_id']==t][met],75)
               -np.percentile(ppo[ppo['task_id']==t][met],25)) for t in TASKS]
    de = [float(np.percentile(dwb[dwb['task_id']==t][met],75)
               -np.percentile(dwb[dwb['task_id']==t][met],25)) for t in TASKS]
    ax.bar(x-w/2, pv, w, label='PPO',      color=PPO_C, alpha=0.85, yerr=pe, capsize=4, error_kw={'lw':1.2})
    ax.bar(x+w/2, dv, w, label='Nav2 DWB', color=DWB_C, alpha=0.85, yerr=de, capsize=4, error_kw={'lw':1.2})
    ax.set_xticks(x); ax.set_xticklabels([TLABELS[t] for t in TASKS], fontsize=8.5)
    ax.set_title(ttl, fontweight="bold"); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_2_per_task_comparison.png', dpi=300, bbox_inches='tight')
plt.close(); print("Fig 4.2 saved")

# Fig 4.3 Noise degradation
fig, axes = plt.subplots(1, 3, figsize=(15,4.5))
fig.suptitle("Figure 4.3 — Sim-to-Real Robustness: Performance vs Sensor Noise", fontsize=12, fontweight="bold")
for i, (met, ttl) in enumerate(zip(['throughput','efficiency','violation_rate'],
                                    ['Throughput Rate','Efficiency Index','Safety Violation Rate'])):
    ax = axes[i]
    for sn, col, ls in [('PPO',PPO_C,'-'),('Nav2_DWB',DWB_C,'--')]:
        sv   = df[df['system']==sn]
        vals = [np.median(sv[sv['noise_sigma']==s][met]) for s in NOISES]
        errs = [float(np.percentile(sv[sv['noise_sigma']==s][met],75)
                     -np.percentile(sv[sv['noise_sigma']==s][met],25)) for s in NOISES]
        ax.plot(NOISES, vals, color=col, ls=ls, lw=2, marker='o', ms=6, label=sn.replace('_',' '))
        ax.fill_between(NOISES,
            np.array(vals)-np.array(errs)*0.5,
            np.array(vals)+np.array(errs)*0.5, alpha=0.12, color=col)
    ax.set_xlabel("Sensor noise σ"); ax.set_title(ttl, fontweight="bold")
    ax.set_xticks(NOISES); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_3_noise_degradation.png', dpi=300, bbox_inches='tight')
plt.close(); print("Fig 4.3 saved")

# Fig 4.4 Robustness heatmap
fig, axes = plt.subplots(1, 2, figsize=(13,4.5))
fig.suptitle("Figure 4.4 — Robustness Heatmap: Efficiency Index", fontsize=12, fontweight="bold")
vmin = np.percentile(df['efficiency'].values, 5)
vmax = np.percentile(df['efficiency'].values, 95)
for sn, ax in zip(['PPO','Nav2_DWB'], axes):
    sv   = df[df['system']==sn]
    grid = np.array([[np.median(sv[(sv['noise_sigma']==s)&(sv['mass_multiplier']==m)]['efficiency'])
                      for m in MASSES] for s in NOISES])
    im = ax.imshow(grid, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(MASSES))); ax.set_xticklabels([f'x{m}' for m in MASSES])
    ax.set_yticks(range(len(NOISES))); ax.set_yticklabels([f's={s}' for s in NOISES])
    ax.set_xlabel("Mass multiplier"); ax.set_ylabel("Sensor noise")
    ax.set_title(sn.replace('_',' '), fontweight="bold")
    for r in range(len(NOISES)):
        for c in range(len(MASSES)):
            ax.text(c, r, f'{grid[r,c]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if grid[r,c]<np.median(grid) else 'black')
    plt.colorbar(im, ax=ax, label='Efficiency Index')
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_4_robustness_heatmap.png', dpi=300, bbox_inches='tight')
plt.close(); print("Fig 4.4 saved")

# Fig 4.5 Distribution boxplots
fig, axes = plt.subplots(1, 4, figsize=(15,5))
fig.suptitle("Figure 4.5 — Distribution Comparison: PPO vs Nav2 DWB", fontsize=12, fontweight="bold")
LBLS = ['Throughput\nRate','Efficiency\nIndex','Violation\nRate','Cycle\nTime']
for i, (met, lbl) in enumerate(zip(METS, LBLS)):
    ax = axes[i]
    bp = ax.boxplot([ppo[met].values, dwb[met].values],
                    patch_artist=True, widths=0.5,
                    medianprops={'color':'black','lw':2},
                    whiskerprops={'lw':1.2}, capprops={'lw':1.2})
    bp['boxes'][0].set_facecolor(PPO_C); bp['boxes'][0].set_alpha(0.75)
    bp['boxes'][1].set_facecolor(DWB_C); bp['boxes'][1].set_alpha(0.75)
    ax.set_xticks([1,2]); ax.set_xticklabels(['PPO','DWB'], fontsize=9)
    ax.set_title(lbl, fontweight="bold")
pp = mpatches.Patch(color=PPO_C, alpha=0.75, label='PPO')
dp = mpatches.Patch(color=DWB_C, alpha=0.75, label='Nav2 DWB')
fig.legend(handles=[pp,dp], loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(0.5,-0.02))
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_5_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close(); print("Fig 4.5 saved")

print(f"\nAll figures saved to {OUT}/")
print("\nFiles generated:")
for f in sorted(os.listdir(OUT)):
    path = f'{OUT}/{f}'
    size = os.path.getsize(path)
    print(f"  {f}  ({size/1024:.0f} KB)")
