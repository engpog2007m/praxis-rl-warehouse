#!/usr/bin/env python3
"""
Train ML Baseline - Script to train traditional ML baselines for comparison.
"""

import argparse
import yaml
import sys
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import WarehouseDataset
from ml.baselines import create_baseline
from utils.logger import setup_logger


def main():
    """Main ML training script."""
    parser = argparse.ArgumentParser(
        description="Train ML baseline for warehouse control comparison"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="supervised",
        choices=["supervised", "random_forest", "svm", "rule_based"],
        help="ML method to use"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of training samples to generate"
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
        name="train_ml",
        log_file=f"{args.log_dir}/train_ml_{args.method}.log"
    )
    
    logger.info(f"Starting ML training with method: {args.method}")
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {args.config}")
        
        # Load or generate dataset
        dataset = WarehouseDataset(config.get("data", {}))
        logger.info("Initialized warehouse dataset")
        
        logger.info(f"Generating {args.num_samples} training samples...")
        X, y = dataset.generate_training_data(num_samples=args.num_samples)
        logger.info(f"Generated training data: X.shape={X.shape}, y.shape={y.shape}")
        
        # Split data
        splits = dataset.split_data(X, y)
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]
        
        logger.info(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")
        
        # Create and train baseline
        baseline_config = config.get("training", {}).get("ml", {})
        baseline_config["seed"] = config.get("project", {}).get("seed", 42)
        
        baseline = create_baseline(args.method, baseline_config)
        logger.info(f"Created {args.method} baseline")
        
        logger.info("Starting training...")
        result = baseline.train(X_train, y_train, X_val, y_val)
        logger.info(f"Training completed: {result}")
        
        # Evaluate on test set
        test_score = baseline.model.score(X_test, y_test)
        logger.info(f"Test set score: {test_score:.4f}")
        
        # Save model
        model_path = f"{args.save_dir}/ml_{args.method}_best.pkl"
        baseline.save(model_path)
        logger.info(f"Saved model to {model_path}")
        
        logger.info("ML training completed successfully!")
        
    except Exception as e:
        logger.error(f"ML training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
