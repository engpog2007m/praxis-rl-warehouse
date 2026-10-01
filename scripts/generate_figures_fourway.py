"""Chapter 4 figures 4.14-4.16 — four-way comparative evaluation."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

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
TASK = {1:'Standard\nBox',2:'Lightweight\nPkg',3:'Heavy\nPallet',
        4:'Fragile\nGlassware',1001:'Heavy+\nFragile'}

df = pd.read_csv(f'{DATA}/four_way_results.csv')
print(f"Loaded {len(df):,} records")

# ---- 4.14 : four-way headline comparison ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax,(m,lab) in zip(axes, [('reward','Episode Reward'),
                             ('efficiency','Efficiency Index'),
                             ('completed','Task Completion Rate')]):
    if m == 'completed':
        vals = [df[df['system']==s]['completed'].mean()*100 for s in ORDER]
        b = ax.bar(range(4), vals, color=[C[s] for s in ORDER],
                   edgecolor='black', lw=0.7)
        for i,v in enumerate(vals):
            ax.text(i, v+1, f'{v:.1f}%', ha='center', fontsize=10)
        ax.set_ylim(0,105); ax.set_ylabel('Completion (%)')
    else:
        data = [df[df['system']==s][m] for s in ORDER]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for p,s in zip(bp['boxes'], ORDER):
            p.set_facecolor(C[s]); p.set_alpha(0.8)
        for med in bp['medians']: med.set_color('black'); med.set_linewidth(2)
        ax.set_ylabel(lab)
    ax.set_xticks(range(1,5) if m!='completed' else range(4))
    ax.set_xticklabels([LBL[s] for s in ORDER], fontsize=9)
    ax.set_title(lab)
fig.suptitle('Figure 4.14 — Four-Way Comparative Evaluation (n = 4,000 episodes)',
             fontweight='bold', fontsize=14)
plt.tight_layout(); plt.savefig(f'{OUT}/fig4_14_four_way_comparison.png',
                                bbox_inches='tight'); plt.close()
print('fig4_14_four_way_comparison.png')

# ---- 4.15 : H3 guided vs unguided across noise ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
noise = sorted(df['noise_sigma'].unique())
for s,mk in [('PPO_guided','o-'),('PPO_unguided','s--')]:
    sub = [df[(df['system']==s)&(df['noise_sigma']==n)]['reward'] for n in noise]
    r  = [x.mean() for x in sub]
    ci = [1.96*x.std()/np.sqrt(len(x)) for x in sub]
    axes[0].plot(noise, r, mk, color=C[s], lw=2.2, ms=8,
                 label=LBL[s].replace('\n',' '))
    axes[0].fill_between(noise, np.array(r)-np.array(ci),
                         np.array(r)+np.array(ci), color=C[s], alpha=0.18)
axes[0].set_xlabel('Sensor noise σ'); axes[0].set_ylabel('Mean episode reward')
axes[0].set_title('Reward vs Sensor Noise'); axes[0].legend()

pv = []
for n in noise:
    g = df[(df['system']=='PPO_guided')&(df['noise_sigma']==n)]['reward']
    u = df[(df['system']=='PPO_unguided')&(df['noise_sigma']==n)]['reward']
    _, p = stats.mannwhitneyu(g, u, alternative='greater'); pv.append(p)
axes[1].bar([f'σ={n}' for n in noise], pv, color='#6B7280',
            edgecolor='black', lw=0.7)
axes[1].axhline(0.05, color='#B03030', ls='--', lw=1.6, label='α = 0.05')
for i,p in enumerate(pv):
    axes[1].text(i, p+0.02, f'{p:.3f}', ha='center', fontsize=10)
axes[1].set_ylabel('Mann-Whitney p-value'); axes[1].set_ylim(0,1.0)
axes[1].set_title('No Significant Difference at Any Noise Level')
axes[1].legend()
fig.suptitle('Figure 4.15 — H3: Guided vs Unguided Policy (null result)',
             fontweight='bold', fontsize=14)
plt.tight_layout(); plt.savefig(f'{OUT}/fig4_15_h3_guided_vs_unguided.png',
                                bbox_inches='tight'); plt.close()
print('fig4_15_h3_guided_vs_unguided.png')

# ---- 4.16 : per-task four-way breakdown ----
fig, ax = plt.subplots(figsize=(12, 5.2))
tasks = [1,2,3,4,1001]; x = np.arange(len(tasks)); w = 0.2
for i,s in enumerate(ORDER):
    vals = [df[(df['system']==s)&(df['task_id']==t)]['completed'].mean()*100
            for t in tasks]
    ax.bar(x + (i-1.5)*w, vals, w, label=LBL[s].replace('\n',' '),
           color=C[s], edgecolor='black', lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([TASK[t] for t in tasks])
ax.set_ylabel('Task Completion Rate (%)'); ax.set_ylim(0,115)
ax.legend(ncol=4, loc='upper center', fontsize=10)
ax.set_title('Figure 4.16 — Completion by Payload Profile, All Four Systems',
             fontweight='bold', fontsize=13)
plt.tight_layout(); plt.savefig(f'{OUT}/fig4_16_four_way_by_task.png',
                                bbox_inches='tight'); plt.close()
print('fig4_16_four_way_by_task.png')
print(f'\nSaved to {OUT}')
