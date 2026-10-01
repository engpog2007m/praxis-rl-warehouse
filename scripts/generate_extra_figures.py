import sys, os
sys.path.insert(0, os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'src'))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data')
OUT  = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'results')
PPO_C = "#2a78d6"; DWB_C = "#eb6834"
TASKS = [1, 2, 3, 4, 1001]
TLABELS = {1:"Standard\nBox",2:"Lightweight\nPkg",3:"Heavy\nPallet",
           4:"Fragile\nGlassware",1001:"Heavy+\nFragile"}

plt.rcParams.update({"font.size":11,"axes.spines.top":False,
    "axes.spines.right":False,"axes.grid":True,
    "grid.alpha":0.3,"grid.linestyle":"--","figure.dpi":300})

df  = pd.read_csv(f'{DATA}/evaluation_results.csv')
ppo = df[df['system']=='PPO']
dwb = df[df['system']=='Nav2_DWB']

# ── Fig 4.6 — Efficiency by task AND mass multiplier (interaction effect) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 4.6 — PPO vs Nav2 DWB: Efficiency Index by Mass Multiplier",
             fontsize=12, fontweight="bold")
MASSES = [1.0, 1.25, 1.50, 2.0]
x = np.arange(len(MASSES)); w = 0.35

for i, (sdf, name, color, ax) in enumerate([
    (ppo,'PPO',PPO_C,axes[0]),(dwb,'Nav2 DWB',DWB_C,axes[1])]):
    for tid in TASKS:
        vals=[np.median(sdf[(sdf['task_id']==tid)&(sdf['mass_multiplier']==m)]['efficiency'])
              for m in MASSES]
        ax.plot(MASSES, vals, marker='o', ms=5, lw=1.8, label=TLABELS[tid].replace('\n',' '))
    ax.set_xlabel("Mass Multiplier"); ax.set_ylabel("Efficiency Index")
    ax.set_title(name, fontweight="bold"); ax.legend(fontsize=8)
    ax.set_xticks(MASSES); ax.set_xticklabels([f'×{m}' for m in MASSES])
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_6_efficiency_by_mass.png', dpi=300, bbox_inches='tight')
plt.close(); print("✅ Fig 4.6 saved — efficiency by mass multiplier")

# ── Fig 4.7 — Reward distribution violin plots ────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 5))
fig.suptitle("Figure 4.7 — Episode Reward Distributions by Task Profile",
             fontsize=12, fontweight="bold")
for i, tid in enumerate(TASKS):
    ax = axes[i]
    ppo_r = ppo[ppo['task_id']==tid]['reward'].values
    dwb_r = dwb[dwb['task_id']==tid]['reward'].values
    vp = ax.violinplot([ppo_r, dwb_r], positions=[1,2], showmedians=True)
    for pc, col in zip(vp['bodies'],[PPO_C,DWB_C]):
        pc.set_facecolor(col); pc.set_alpha(0.7)
    vp['cmedians'].set_color('black'); vp['cmedians'].set_linewidth(2)
    ax.set_xticks([1,2]); ax.set_xticklabels(['PPO','DWB'], fontsize=9)
    ax.set_title(TLABELS[tid], fontweight="bold", fontsize=9)
    if i == 0: ax.set_ylabel("Episode Reward")
pp = mpatches.Patch(color=PPO_C,alpha=0.7,label='PPO')
dp = mpatches.Patch(color=DWB_C,alpha=0.7,label='Nav2 DWB')
fig.legend(handles=[pp,dp],loc='lower center',ncol=2,fontsize=10,bbox_to_anchor=(0.5,-0.02))
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_7_reward_distributions.png', dpi=300, bbox_inches='tight')
plt.close(); print("✅ Fig 4.7 saved — reward distributions by task")

# ── Fig 4.8 — Dataset integration architecture diagram ────────────────────
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis('off')
fig.suptitle("Figure 4.8 — Dataset Integration Architecture: DataRegistry Pipeline",
             fontsize=12, fontweight="bold")

boxes = [
    (1.0, 4.2, 2.0, 0.9, '#1F3864', 'white', 'Mutschler Lab\nWarehouse Dataset\n36.6m × 36.0m'),
    (1.0, 2.8, 2.0, 0.9, '#2E75B6', 'white', 'SKU110K Dataset\n11,762 images\nmean 147 obj/image'),
    (1.0, 1.4, 2.0, 0.9, '#0A7C8A', 'white', 'YCB Object Set\n4 objects\nmeasured mass/friction'),
    (4.5, 2.5, 2.2, 1.2, '#1A5C2A', 'white', 'DataRegistry\n(runtime loader)'),
    (7.5, 4.2, 2.2, 0.9, '#7A4A00', 'white', 'Arena Geometry\n36.6 × 36.0m bounds'),
    (7.5, 2.8, 2.2, 0.9, '#7A4A00', 'white', 'Noise Sigmas\nσ∈{0.00,0.05,0.15,0.30}'),
    (7.5, 1.4, 2.2, 0.9, '#7A4A00', 'white', 'Task Registry\n0.098–0.514 kg masses'),
    (10.2, 2.8, 1.5, 1.2, '#1F3864', 'white', 'Warehouse\nEnv\n(PPO Training)'),
]
for (x,y,w,h,fc,tc,label) in boxes:
    rect = plt.Rectangle((x,y),w,h,facecolor=fc,edgecolor='white',linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',color=tc,fontsize=8,fontweight='bold')

arrows = [(3.0,4.65,4.5,3.1),(3.0,3.25,4.5,3.1),(3.0,1.85,4.5,3.1),
          (6.7,3.7,7.5,4.65),(6.7,3.1,7.5,3.25),(6.7,2.5,7.5,1.85),
          (9.7,3.4,10.2,3.4)]
for (x1,y1,x2,y2) in arrows:
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->',color='#555555',lw=1.5))

plt.tight_layout()
plt.savefig(f'{OUT}/fig4_8_dataset_integration.png', dpi=300, bbox_inches='tight')
plt.close(); print("✅ Fig 4.8 saved — dataset integration diagram")

print(f"\n✅ All extra figures saved to {OUT}/")
