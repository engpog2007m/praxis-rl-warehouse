#!/usr/bin/env python3
import time
from stable_baselines3 import PPO
from src.simulation.warehouse_env import WarehouseEnv

def evaluate_model(model_path):
    # Initialize environment with human render mode
    env = WarehouseEnv(render_mode="human")
    
    # Load the trained agent
    print(f"[INFO] Loading model from: {model_path}")
    model = PPO.load(model_path, env=env)
    
    obs, _ = env.reset()
    done = False
    
    print("[INFO] Starting visualization... Press Ctrl+C to stop.")
    try:
        while not done:
            # Predict the next best action based on the learned policy
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Slow down slightly for human viewing
            time.sleep(0.05)
            
            done = terminated or truncated
            if done:
                print("[INFO] Episode finished. Resetting...")
                obs, _ = env.reset()
                done = False
    except KeyboardInterrupt:
        print("[INFO] Visualization stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    # Point to the model we saved during the training run
    model_path = "data/trained_models/ppo_final"
    evaluate_model(model_path)
