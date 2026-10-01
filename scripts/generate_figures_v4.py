"""
Chapter 4 Figures — Gazebo v4 Physics Layer
Generates figures from the final corrected-baseline evaluation.
Empty/uninformative panels removed.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

DATA = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data')
OUT  = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'results')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.size': 11, 'axes.titlesize': 13,
    'axes.titleweight': 'bold', 'axes.grid': True,
    'grid.alpha': 0.3, 'grid.linestyle': '--',
})

PPO_C, DWB_C = '#3B7DD8', '#E8823C'
TASK_LABEL = {1:'Standard\nBox', 2:'Lightweight\nPkg', 3:'Heavy\nPallet',
              4:'Fragile\nGlassware', 1001:'Heavy+\nFragile'}

df = pd.read_csv(f'{DATA}/gazebo_trained_results_v4.csv')
ppo = df[df['system']=='PPO_gazebo']
dwb = df[df['system']=='Nav2_DWB']
print(f"Loaded {len(df):,} records")

# ---------- Fig 4.9: Headline comparison ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
metrics = [('reward','Episode Reward'), ('efficiency','Efficiency Index'),
           ('cycle_time','Cycle Time (ticks)')]
for ax,(m,lab) in zip(axes, metrics):
    bp = ax.boxplot([ppo[m], dwb[m]], labels=['PPO','Nav2 DWB'],
                    patch_artist=True, widths=0.55, showfliers=False)
    for patch,c in zip(bp['boxes'], [PPO_C, DWB_C]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    for med in bp['medians']:
        med.set_color('black'); med.set_linewidth(2)
    _,p = stats.mannwhitneyu(ppo[m], dwb[m], alternative='two-sided')
    ax.set_title(f'{lab}\n(p = {p:.4f})')
    ax.set_ylabel(lab)
fig.suptitle('Figure 4.9 — Gazebo Physics Layer: PPO vs Nav2 DWB (n=2,000)',
             fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_9_gazebo_headline.png', bbox_inches='tight')
plt.close()
print("fig4_9_gazebo_headline.png")

# ---------- Fig 4.10: Completion by task ----------
fig, ax = plt.subplots(figsize=(10, 5))
tasks = [1,2,3,4,1001]
x = np.arange(len(tasks)); w = 0.36
pv = [ppo[ppo['task_id']==t]['completed'].mean()*100 for t in tasks]
dv = [dwb[dwb['task_id']==t]['completed'].mean()*100 for t in tasks]
b1 = ax.bar(x-w/2, pv, w, label='PPO', color=PPO_C, edgecolor='black', lw=0.6)
b2 = ax.bar(x+w/2, dv, w, label='Nav2 DWB', color=DWB_C, edgecolor='black', lw=0.6)
for bars in (b1,b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.5,
                f'{b.get_height():.0f}%', ha='center', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([TASK_LABEL[t] for t in tasks])
ax.set_ylabel('Task Completion Rate (%)'); ax.set_ylim(0,115)
ax.legend(); ax.axhline(100, color='grey', ls=':', lw=1)
ax.text(0.02, 101.5, 'ceiling', transform=ax.get_yaxis_transform(),
        fontsize=8, color='grey')
ax.set_title('Figure 4.10 — Task Completion Rate by Payload Profile',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_10_completion_by_task.png', bbox_inches='tight')
plt.close()
print("fig4_10_completion_by_task.png")

# ---------- Fig 4.11: Robustness vs sensor noise ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
noise = sorted(df['noise_sigma'].unique())
pc = [ppo[ppo['noise_sigma']==s]['completed'].mean()*100 for s in noise]
dc = [dwb[dwb['noise_sigma']==s]['completed'].mean()*100 for s in noise]
axes[0].plot(noise, pc, 'o-', color=PPO_C, lw=2.2, ms=8, label='PPO')
axes[0].plot(noise, dc, 's--', color=DWB_C, lw=2.2, ms=8, label='Nav2 DWB')
axes[0].set_xlabel('Sensor noise σ'); axes[0].set_ylabel('Completion Rate (%)')
axes[0].set_title('Completion vs Sensor Noise'); axes[0].legend()

gap = [p-d for p,d in zip(pc,dc)]
axes[1].bar([f'σ={s}' for s in noise], gap, color='#2E7D5B',
            edgecolor='black', lw=0.6)
for i,g in enumerate(gap):
    axes[1].text(i, g+0.15, f'+{g:.1f}', ha='center', fontsize=10)
axes[1].set_ylabel('PPO advantage (percentage points)')
axes[1].set_title('PPO Advantage Widens Under Noise')
fig.suptitle('Figure 4.11 — Robustness to Sensor Uncertainty (SKU110K-calibrated)',
             fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_11_noise_robustness.png', bbox_inches='tight')
plt.close()
print("fig4_11_noise_robustness.png")

# ---------- Fig 4.12: Reward distribution by task ----------
fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))
for ax,t in zip(axes, tasks):
    pp = ppo[ppo['task_id']==t]['reward']
    dd = dwb[dwb['task_id']==t]['reward']
    parts = ax.violinplot([pp,dd], positions=[1,2], showmedians=True, widths=0.8)
    for pc_,c in zip(parts['bodies'], [PPO_C, DWB_C]):
        pc_.set_facecolor(c); pc_.set_alpha(0.7)
    ax.set_xticks([1,2]); ax.set_xticklabels(['PPO','DWB'])
    ax.set_title(TASK_LABEL[t], fontsize=11)
    if t==1: ax.set_ylabel('Episode Reward')
fig.suptitle('Figure 4.12 — Reward Distributions by Payload Profile (Gazebo v4)',
             fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_12_reward_by_task.png', bbox_inches='tight')
plt.close()
print("fig4_12_reward_by_task.png")

# ---------- Fig 4.13: Reward function correction (v3 vs v4) ----------
try:
    v3 = pd.read_csv(f'{DATA}/gazebo_training_curves_v3.csv')
    v4 = pd.read_csv(f'{DATA}/gazebo_training_curves_v4.csv')
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(v3['timestep'], v3['mean_reward'], color='#C0392B',
            lw=1.2, alpha=0.85, label='v3 — original C_safety = −2.0')
    ax.plot(v4['timestep'], v4['mean_reward'], color='#2E7D5B',
            lw=1.2, alpha=0.85, label='v4 — rebalanced C_safety = −0.5')
    ax.axhline(0, color='black', ls='--', lw=1)
    ax.set_xlabel('Training timesteps'); ax.set_ylabel('Mean episode reward')
    ax.legend(loc='lower left')
    ax.set_title('Figure 4.13 — Effect of Reward Function Correction',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig4_13_reward_correction.png', bbox_inches='tight')
    plt.close()
    print("fig4_13_reward_correction.png")
except Exception as e:
    print(f"skipped fig4_13: {e}")

print(f"\nSaved to {OUT}")
