#!/usr/bin/env python3
"""
Train RL Agent with Mutchler Lab Task Orchestration - Advanced training script.

This script trains RL agents using the Mutchler lab warehouse dataset
with realistic task orchestration for multi-robot coordination.
"""

import argparse
import yaml
import sys
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.orchestrated_env import OrchestrationAwareEnv
from data.mutchler_lab_loader import MutchlerLabDataset, TaskOrchestrator
from rl.agents import create_agent
from utils.logger import setup_logger


def main():
    """Main training script with task orchestration."""
    parser = argparse.ArgumentParser(
        description="Train RL agent with Mutchler lab task orchestration"
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
        "--warehouse",
        type=str,
        default="medium_warehouse",
        choices=["small_warehouse", "medium_warehouse", "large_warehouse"],
        help="Warehouse layout to use"
    )
    parser.add_argument(
        "--task-set",
        type=str,
        default="default_task_set",
        help="Task set to use"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100000,
        help="Number of training steps"
    )
    parser.add_argument(
        "--num-robots",
        type=int,
        default=1,
        help="Number of robots in warehouse"
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
        name="train_rl_orchestrated",
        log_file=f"{args.log_dir}/train_rl_orchestrated_{args.algorithm}.log"
    )
    
    logger.info(f"Starting orchestrated training with algorithm: {args.algorithm}")
    logger.info(f"Warehouse: {args.warehouse}, Task Set: {args.task_set}, Robots: {args.num_robots}")
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {args.config}")
        
        # Initialize Mutchler lab dataset
        mutchler_config = config.get("data", {}).get("mutchler_lab", {})
        dataset = MutchlerLabDataset(
            config.get("data", {}).get("mutchler_lab_path", "./data/mutchler_lab")
        )
        logger.info("Loaded Mutchler lab dataset")
        
        # Get warehouse layout
        warehouse_layout = dataset.get_warehouse_layout(args.warehouse)
        if not warehouse_layout:
            logger.error(f"Warehouse layout '{args.warehouse}' not found")
            sys.exit(1)
        
        logger.info(f"Loaded warehouse layout: {args.warehouse}")
        
        # Create task orchestrator
        orchestrator = TaskOrchestrator(dataset, num_robots=args.num_robots)
        orchestrator.load_tasks(args.task_set)
        logger.info(f"Created task orchestrator with {args.num_robots} robots")
        
        # Create environment with task orchestration
        env = OrchestrationAwareEnv(
            task_orchestrator=orchestrator,
            num_agents=args.num_robots,
            warehouse_config=config.get("simulation", {}),
            max_episode_steps=2000,
        )
        logger.info("Created orchestration-aware warehouse environment")
        
        # Create agent
        agent = create_agent(args.algorithm, env, config.get("training", {}))
        logger.info(f"Created {args.algorithm.upper()} agent")
        
        # Train agent
        logger.info(f"Starting training for {args.num_steps} steps...")
        logger.info(f"Tasks to complete: {orchestrator.metrics['total_tasks']}")
        
        result = agent.train(num_steps=args.num_steps)
        logger.info(f"Training completed: {result}")
        
        # Save model
        model_path = f"{args.save_dir}/rl_{args.algorithm}_orchestrated_best"
        agent.save(model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Print final statistics
        logger.info("\n" + "="*80)
        logger.info("FINAL STATISTICS")
        logger.info("="*80)
        orchestrator.print_status()
        
        logger.info("Orchestrated training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
