"""Chapter 4 figures 4.17-4.18 — LiDAR/peer-obstacle era results."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data')
OUT  = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'results')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'figure.dpi':300,'savefig.dpi':300,'font.size':11,
    'axes.titlesize':13,'axes.titleweight':'bold','axes.grid':True,
    'grid.alpha':0.3,'grid.linestyle':'--'})

C = {'PPO_guided':'#2E5EAA','PPO_unguided':'#7BA7D9',
     'BC_ml':'#C4622D','Nav2_DWB':'#6B7280'}
LBL = {'PPO_guided':'PPO\n(guided)','PPO_unguided':'PPO\n(unguided)',
       'BC_ml':'Behavioural\nCloning','Nav2_DWB':'Nav2 DWB'}
ORDER = ['PPO_guided','PPO_unguided','BC_ml','Nav2_DWB']

# ---- 4.17 : outcome breakdown (completed / collided / timed_out) ----
df = pd.read_csv(f'{DATA}/four_way_lidar_results.csv')
print(f"Loaded {len(df):,} records")

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(ORDER))
w = 0.25

completed = [df[df['system']==s]['completed'].mean()*100 for s in ORDER]
collided  = [df[df['system']==s]['collided'].mean()*100 for s in ORDER]
timed_out = [df[df['system']==s]['timed_out'].mean()*100 for s in ORDER]

b1 = ax.bar(x - w, completed, w, label='Completed', color='#2C7A4B',
            edgecolor='black', lw=0.6)
b2 = ax.bar(x, collided, w, label='Collided', color='#B03030',
            edgecolor='black', lw=0.6)
b3 = ax.bar(x + w, timed_out, w, label='Timed out', color='#8C8C8C',
            edgecolor='black', lw=0.6)

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+1, f'{h:.1f}%',
                ha='center', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([LBL[s] for s in ORDER])
ax.set_ylabel('Episode outcome rate (%)')
ax.set_ylim(0, 65)
ax.legend(loc='upper left')
ax.set_title('Figure 4.17 — Episode Outcomes with Active Peer Obstacle\n'
              '(n = 800 episodes per system)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_17_outcomes_peer_obstacle.png', bbox_inches='tight')
plt.close()
print('fig4_17_outcomes_peer_obstacle.png')

# ---- 4.18 : H3 safety-effectiveness tradeoff, guided vs unguided ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

pairs = [('PPO_guided', C['PPO_guided']), ('PPO_unguided', C['PPO_unguided'])]

# left: completion vs collision scatter-style paired bars
metrics = ['completed', 'collided']
mlabels = ['Completion', 'Collision']
xpos = np.arange(len(metrics))
ww = 0.35
for i, (sys_name, col) in enumerate(pairs):
    vals = [df[df['system']==sys_name][m].mean()*100 for m in metrics]
    bars = axes[0].bar(xpos + (i-0.5)*ww, vals, ww,
                       label=LBL[sys_name].replace('\n',' '),
                       color=col, edgecolor='black', lw=0.6)
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x()+bar.get_width()/2, v+1.5, f'{v:.1f}%',
                    ha='center', fontsize=10)
axes[0].set_xticks(xpos)
axes[0].set_xticklabels(mlabels)
axes[0].set_ylabel('Rate (%)')
axes[0].set_ylim(0, 65)
axes[0].legend()
axes[0].set_title('Completion vs Collision Rate')

# right: significance annotations
axes[1].axis('off')
txt = (
    "H3 — Guided vs Unguided\n"
    "(n = 800 per arm)\n\n"
    f"Completion:  29.5%  vs  56.3%\n"
    f"  Mann-Whitney p < 0.0001\n\n"
    f"Collision:   18.2%  vs  37.4%\n"
    f"  Mann-Whitney p < 0.0001\n"
    f"  effect size r = +0.191\n\n"
    "Guidance mechanisms (payload-adaptive\n"
    "reward shaping + domain randomisation)\n"
    "significantly reduce collision rate but\n"
    "significantly reduce task completion.\n\n"
    "H3 is partially supported: the safety\n"
    "claim holds; the effectiveness claim\n"
    "does not."
)
axes[1].text(0.02, 0.98, txt, transform=axes[1].transAxes,
             fontsize=11.5, va='top', ha='left', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#F2F2F2', edgecolor='#999'))

fig.suptitle('Figure 4.18 — H3: Safety-Effectiveness Tradeoff',
             fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_18_h3_tradeoff.png', bbox_inches='tight')
plt.close()
print('fig4_18_h3_tradeoff.png')

# ---- 4.19 : mobility-floor before/after correction ----
fig, ax = plt.subplots(figsize=(9, 5.5))
labels = ['Pre-correction\n(no wheel friction,\nchassis on floor)',
          'Post-correction\n(mu=1.2 friction,\nchassis clearance)']
displacement_mm = [0.5, 1750]  # 0.5mm vs 1.75m over comparable window
colors_bar = ['#B03030', '#2C7A4B']

bars = ax.bar(labels, displacement_mm, color=colors_bar,
             edgecolor='black', lw=0.8, width=0.5)
ax.set_yscale('log')
ax.set_ylabel('Measured displacement (mm, log scale)')
for bar, val in zip(bars, displacement_mm):
    label = f'{val:.1f} mm' if val < 100 else f'{val/1000:.2f} m'
    ax.text(bar.get_x()+bar.get_width()/2, val*1.3, label,
            ha='center', fontsize=11, fontweight='bold')
ax.set_title('Figure 4.19 — Physics Defect: Commanded vs Measured Motion\n'
              '(full commanded velocity, ~3-4 s window)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_19_physics_correction.png', bbox_inches='tight')
plt.close()
print('fig4_19_physics_correction.png')

print(f'\nSaved to {OUT}')
