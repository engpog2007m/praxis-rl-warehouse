#!/usr/bin/env python3
"""
Train RL Agent - Script to train RL agents for warehouse control.
"""

import argparse
import yaml
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.warehouse_env import WarehouseEnv
from rl.agents import create_agent
from utils.logger import setup_logger


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train RL agent for warehouse control"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "dqn", "a3c", "ddpg"],
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100000,
        help="Number of training steps"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/trained_models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save logs"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(
        name="train_rl",
        log_file=f"{args.log_dir}/train_rl_{args.algorithm}.log"
    )
    
    logger.info(f"Starting training with algorithm: {args.algorithm}")
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {args.config}")
        
        # Create environment
        env = WarehouseEnv(
            num_agents=1,
            warehouse_config=config.get("simulation", {})
        )
        logger.info("Created warehouse environment")
        
        # Create agent
        agent = create_agent(args.algorithm, env, config.get("training", {}))
        logger.info(f"Created {args.algorithm.upper()} agent")
        
        # Train agent
        logger.info(f"Starting training for {args.num_steps} steps...")
        result = agent.train(num_steps=args.num_steps)
        logger.info(f"Training completed: {result}")
        
        # Save model
        model_path = f"{args.save_dir}/rl_{args.algorithm}_best"
        agent.save(model_path)
        logger.info(f"Saved model to {model_path}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
