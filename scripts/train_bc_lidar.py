"""
Behavioural cloning baseline — supervised regression from observation
to velocity command, trained on Nav2 DWB demonstrations.

Two model families are fitted and compared on held-out data:
  - Random Forest regressor  (ensemble, non-parametric)
  - Multi-layer perceptron   (parametric, matches PPO's policy capacity)

The better-performing model is retained as the traditional-ML comparator.
"""
import os, json, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

DATA = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data/dwb_lidar_demonstrations.csv')
OUT  = os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'models')
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)
print(f"Loaded {len(df):,} state-action pairs")

OBS = [f'o{i}' for i in range(31)]
ACT = ['a0','a1']

X = df[OBS].values.astype(np.float32)
y = df[ACT].values.astype(np.float32)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

results = {}

print("\nFitting Random Forest...", flush=True)
rf = RandomForestRegressor(n_estimators=120, max_depth=18,
                           min_samples_leaf=4, n_jobs=-1, random_state=42)
rf.fit(Xtr, ytr)
rf_pred = rf.predict(Xte)
results['random_forest'] = {
    'mse': float(mean_squared_error(yte, rf_pred)),
    'r2':  float(r2_score(yte, rf_pred)),
}
print(f"  MSE {results['random_forest']['mse']:.6f}   "
      f"R2 {results['random_forest']['r2']:.4f}")

print("\nFitting MLP...", flush=True)
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='tanh',
                   learning_rate_init=1e-3, max_iter=400,
                   early_stopping=True, n_iter_no_change=20,
                   random_state=42)
mlp.fit(Xtr_s, ytr)
mlp_pred = mlp.predict(Xte_s)
results['mlp'] = {
    'mse': float(mean_squared_error(yte, mlp_pred)),
    'r2':  float(r2_score(yte, mlp_pred)),
}
print(f"  MSE {results['mlp']['mse']:.6f}   R2 {results['mlp']['r2']:.4f}")

best = 'random_forest' if results['random_forest']['r2'] >= results['mlp']['r2'] else 'mlp'
print(f"\nRetained model: {best}")

bundle = {
    'model_type': best,
    'model': rf if best == 'random_forest' else mlp,
    'scaler': scaler if best == 'mlp' else None,
    'obs_cols': OBS, 'act_cols': ACT,
    'metrics': results,
    'n_train': int(len(Xtr)), 'n_test': int(len(Xte)),
}
with open(f'{OUT}/bc_lidar_policy.pkl', 'wb') as f:
    pickle.dump(bundle, f)

with open(os.path.join(os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis')), 'data/bc_lidar_metrics.json'), 'w') as f:
    json.dump({'best': best, 'results': results,
               'n_train': int(len(Xtr)), 'n_test': int(len(Xte))}, f, indent=2)

print(f"Saved -> {OUT}/bc_lidar_policy.pkl")
