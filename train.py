import os
import yaml
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from src.simulation.warehouse_env import WarehouseEnv

def train():
    # Load config
    with open("src/config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # 1. Standardized Environment Wrapper
    log_dir = "data/logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Using a vectorized environment for speed
    env = make_vec_env(lambda: WarehouseEnv(), n_envs=4)
    env = Monitor(env, log_dir) 

    # 2. Agent Initialization
    model = PPO("MlpPolicy", env, verbose=1, **config['training'])

    # 3. Training Loop
    print("[INFO] Starting PPO training...")
    model.learn(total_timesteps=50000)

    # 4. Save Final Results
    model.save("data/trained_models/ppo_final")
    
    # The monitor file contains the precise episodic data needed for thesis plots
    # You can load this file into any plotting tool
    print(f"[INFO] Training complete. Monitor data saved at {log_dir}")

if __name__ == "__main__":
    train()
