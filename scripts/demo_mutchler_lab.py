#!/usr/bin/env python3
"""
Mutchler Lab Dataset and Task Orchestration Demo - Demonstrate usage of dataset and orchestration.

This script demonstrates how to:
1. Load Mutchler lab warehouse datasets
2. Initialize task orchestration
3. Assign and manage tasks
4. Evaluate performance with orchestration
"""

import sys
import argparse
from pathlib import Path
import tempfile
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.mutchler_lab_loader import MutchlerLabDataset, TaskOrchestrator
from src.utils.logger import setup_logger


def demo_dataset_operations(dataset: MutchlerLabDataset):
    """Demonstrate dataset operations."""
    print("\n" + "="*80)
    print("MUTCHLER LAB DATASET OPERATIONS")
    print("="*80)
    
    # List available warehouses
    print("\nAvailable Warehouse Layouts:")
    for warehouse_id in dataset.list_warehouses():
        print(f"  - {warehouse_id}")
    
    # Load and display warehouse details
    warehouse = dataset.get_warehouse_layout("medium_warehouse")
    print(f"\nMedium Warehouse Configuration:")
    print(f"  Dimensions: {warehouse['dimensions']['x']}m x {warehouse['dimensions']['y']}m")
    print(f"  Number of Shelves: {warehouse['num_shelves']}")
    print(f"  Storage Locations: {len(warehouse['storage_locations'])}")
    print(f"  Packing Station: {warehouse['packing_station']}")
    
    # List available task sets
    print(f"\nAvailable Task Sets:")
    for task_set_id in dataset.list_task_sets():
        tasks = dataset.get_task_set(task_set_id)
        print(f"  - {task_set_id}: {len(tasks)} tasks")
    
    # Analyze first task
    tasks = dataset.get_task_set("default_task_set")
    if tasks:
        task = tasks[0]
        print(f"\nSample Task Details (Task ID: {task.task_id}):")
        print(f"  Order ID: {task.order_id}")
        print(f"  Priority: {task.priority}")
        print(f"  Pick Tasks: {len(task.pick_tasks)}")
        for i, pick_task in enumerate(task.pick_tasks[:3]):
            print(f"    {i+1}. {pick_task.item_sku} from {pick_task.source_location}")
        print(f"  Delivery Location: {task.place_task.destination}")


def demo_task_orchestration(dataset: MutchlerLabDataset, num_robots: int = 2):
    """Demonstrate task orchestration."""
    print("\n" + "="*80)
    print("TASK ORCHESTRATION DEMO")
    print("="*80)
    
    # Create orchestrator
    orchestrator = TaskOrchestrator(dataset, num_robots=num_robots)
    orchestrator.load_tasks("default_task_set")
    
    print(f"\nInitialized Orchestrator:")
    print(f"  Number of Robots: {num_robots}")
    print(f"  Total Tasks: {orchestrator.metrics['total_tasks']}")
    
    # Simulate task execution
    print(f"\nSimulating Task Execution ({num_robots} robots):")
    print("-"*80)
    
    step = 0
    max_steps = 50  # Simulate 50 task assignments/completions
    
    while step < max_steps and (orchestrator.task_queue or orchestrator.active_tasks):
        # Assign tasks to available robots
        for robot_id in range(num_robots):
            if orchestrator.robot_states[robot_id]["available"] and orchestrator.task_queue:
                task = orchestrator.assign_next_task(robot_id)
                if task:
                    print(f"  Step {step}: Assigned {task.task_id} to Robot {robot_id}")
                    
                    # Simulate robot working on task (random success)
                    import random
                    success = random.random() > 0.05  # 95% success rate
                    
                    # Complete task
                    orchestrator.complete_task(robot_id, success=success)
                    if success:
                        print(f"    ✓ Task completed successfully")
                    else:
                        print(f"    ✗ Task failed")
                    
                    step += 1
        
        if not orchestrator.robot_states[0]["available"]:
            # Wait for a robot to become available
            break
    
    # Print final statistics
    orchestrator.print_status()


def demo_custom_workflow():
    """Demonstrate custom workflow with Mutchler lab data."""
    print("\n" + "="*80)
    print("CUSTOM WORKFLOW EXAMPLE")
    print("="*80)
    
    print("""
This example shows how to integrate Mutchler lab dataset with RL training:

1. Load Mutchler Lab Dataset
   ├─ Load warehouse layouts
   ├─ Load task specifications
   └─ Load performance benchmarks

2. Create Task Orchestrator
   ├─ Initialize with dataset
   ├─ Assign robots
   └─ Load task set

3. Create Environment with Orchestration
   ├─ Use OrchestrationAwareEnv
   ├─ Provide task orchestrator
   └─ Environment handles task coordination

4. Train RL Agent
   ├─ Learn warehouse task policies
   ├─ Optimize multi-robot coordination
   └─ Track performance metrics

5. Evaluate Performance
   ├─ Compare against baselines
   ├─ Measure task completion rates
   └─ Analyze efficiency gains

Example Training Script:
    python scripts/train_rl_orchestrated.py \\
        --algorithm ppo \\
        --warehouse medium_warehouse \\
        --num-robots 2 \\
        --num-steps 100000
    """)


def main():
    """Main demonstration."""
    parser = argparse.ArgumentParser(
        description="Demonstrate Mutchler Lab dataset and task orchestration"
    )
    parser.add_argument(
        "--demo",
        type=str,
        default="all",
        choices=["dataset", "orchestration", "workflow", "all"],
        help="Which demo to run"
    )
    parser.add_argument(
        "--num-robots",
        type=int,
        default=2,
        help="Number of robots for orchestration demo"
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
        name="mutchler_demo",
        log_file=f"{args.log_dir}/mutchler_demo.log"
    )
    
    try:
        # Create temporary data directory for demo
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = MutchlerLabDataset(tmpdir)
            
            # Run selected demos
            if args.demo in ["dataset", "all"]:
                demo_dataset_operations(dataset)
            
            if args.demo in ["orchestration", "all"]:
                demo_task_orchestration(dataset, num_robots=args.num_robots)
            
            if args.demo in ["workflow", "all"]:
                demo_custom_workflow()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nNext Steps:")
        print("1. Explore the Mutchler lab dataset documentation")
        print("2. Run the orchestrated training script")
        print("3. Evaluate performance with real datasets")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
