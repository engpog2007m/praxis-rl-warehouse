#!/usr/bin/env python3
"""
Enhanced Warehouse Environment with Mutchler Lab Task Orchestration.

Integrates task orchestration with the warehouse environment for
realistic multi-task, multi-robot coordination scenarios.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, List, Optional

from .warehouse_env import WarehouseEnv
from src.data.mutchler_lab_loader import TaskOrchestrator, WarehouseTask


class OrchestrationAwareEnv(WarehouseEnv):
    """
    Warehouse environment with integrated task orchestration.
    
    Extends WarehouseEnv to include:
    - Task orchestration using Mutchler lab datasets
    - Multi-task reward signals
    - Realistic task scheduling
    - Performance metrics aligned with warehouse KPIs
    """
    
    def __init__(
        self,
        task_orchestrator: TaskOrchestrator,
        num_agents: int = 1,
        warehouse_config: Dict[str, Any] = None,
        render_mode: str = None,
        max_episode_steps: int = 2000,
    ):
        """
        Initialize orchestration-aware environment.
        
        Args:
            task_orchestrator: TaskOrchestrator instance
            num_agents: Number of robots
            warehouse_config: Warehouse configuration
            render_mode: Rendering mode
            max_episode_steps: Max steps per episode
        """
        super().__init__(
            num_agents=num_agents,
            warehouse_config=warehouse_config,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
        
        self.orchestrator = task_orchestrator
        self.current_tasks: Dict[int, WarehouseTask] = {}
        self.task_progress: Dict[int, Dict[str, Any]] = {}
        
        # Extended observation space to include task info
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(420,),  # 370 (original) + 50 (task info)
            dtype=np.float32,
        )
    
    def reset(
        self, seed: int = None, options: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and load tasks."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Clear previous task state
        self.current_tasks.clear()
        self.task_progress.clear()
        
        # Assign initial tasks to robots
        for robot_id in range(self.num_agents):
            task = self.orchestrator.assign_next_task(robot_id)
            if task:
                self.current_tasks[robot_id] = task
                self.task_progress[robot_id] = {
                    "pick_index": 0,
                    "items_picked": [],
                    "arrived_at_destination": False,
                }
        
        # Include task info in observation
        obs = self._extend_observation(obs)
        
        info["tasks_assigned"] = len(self.current_tasks)
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute environment step with task orchestration.
        
        Args:
            action: Robot action [linear_vel, angular_vel, gripper_cmd]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Process task-related rewards and updates
        task_reward = self._process_tasks()
        reward += task_reward
        
        # Check for task completion
        for robot_id in range(self.num_agents):
            if robot_id in self.current_tasks:
                if self._check_task_complete(robot_id):
                    self._complete_task(robot_id)
                    
                    # Assign next task
                    next_task = self.orchestrator.assign_next_task(robot_id)
                    if next_task:
                        self.current_tasks[robot_id] = next_task
                        self.task_progress[robot_id] = {
                            "pick_index": 0,
                            "items_picked": [],
                            "arrived_at_destination": False,
                        }
        
        # Update info with task status
        info.update(self._get_task_info())
        
        # Extend observation with task info
        obs = self._extend_observation(obs)
        
        return obs, reward, terminated, truncated, info
    
    def _process_tasks(self) -> float:
        """
        Process task-related rewards and state updates.
        
        Returns:
            Task-related reward component
        """
        reward = 0.0
        
        for robot_id in range(self.num_agents):
            if robot_id not in self.current_tasks:
                continue
            
            task = self.current_tasks[robot_id]
            progress = self.task_progress[robot_id]
            robot_pos = self.robot_states[robot_id].position[:2]
            
            pick_idx = progress["pick_index"]
            
            # Process pick tasks
            if pick_idx < len(task.pick_tasks):
                pick_task = task.pick_tasks[pick_idx]
                target_loc = np.array(pick_task.source_location)
                distance_to_pick = np.linalg.norm(robot_pos - target_loc)
                
                # Reward for approaching pick location
                if distance_to_pick < 0.5:
                    reward += 1.0
                    
                    # Check if robot can pick
                    if self.robot_states[robot_id].gripper_state > 0.8:
                        progress["items_picked"].append(pick_task.item_sku)
                        progress["pick_index"] += 1
                        reward += 5.0
            
            # Process delivery task
            else:
                delivery_loc = np.array(task.place_task.destination)
                distance_to_delivery = np.linalg.norm(robot_pos - delivery_loc)
                
                # Reward for approaching delivery location
                if distance_to_delivery < 1.0:
                    reward += 1.0
                    progress["arrived_at_destination"] = True
        
        return reward
    
    def _check_task_complete(self, robot_id: int) -> bool:
        """Check if robot's current task is complete."""
        if robot_id not in self.current_tasks:
            return False
        
        task = self.current_tasks[robot_id]
        progress = self.task_progress[robot_id]
        
        # Task complete if all picks are done and delivery location reached
        return (
            progress["pick_index"] >= len(task.pick_tasks) and
            progress["arrived_at_destination"]
        )
    
    def _complete_task(self, robot_id: int):
        """Mark task as complete in orchestrator."""
        if robot_id in self.current_tasks:
            task = self.current_tasks.pop(robot_id)
            self.orchestrator.complete_task(robot_id, success=True)
    
    def _extend_observation(self, base_obs: np.ndarray) -> np.ndarray:
        """Extend base observation with task information."""
        task_info = np.zeros(50)
        
        if len(self.current_tasks) > 0:
            # Add task-related features
            for i, (robot_id, task) in enumerate(self.current_tasks.items()):
                if i >= 5:  # Max 5 tasks in observation
                    break
                
                progress = self.task_progress[robot_id]
                offset = i * 10
                
                # Encode task priority
                task_info[offset + 0] = task.priority / 3.0
                
                # Encode pick progress
                task_info[offset + 1] = progress["pick_index"] / max(len(task.pick_tasks), 1)
                
                # Encode deadline urgency
                deadline = task.place_task.deadline or 3600.0
                task_info[offset + 2] = min(self.current_step / deadline, 1.0)
                
                # Encode destination direction (from robot position)
                if robot_id < len(self.robot_states):
                    robot_pos = self.robot_states[robot_id].position[:2]
                    dest = np.array(task.place_task.destination)
                    direction = dest - robot_pos
                    dist = np.linalg.norm(direction)
                    
                    if dist > 0:
                        task_info[offset + 3] = direction[0] / dist
                        task_info[offset + 4] = direction[1] / dist
                        task_info[offset + 5] = min(dist / 50.0, 1.0)
        
        extended_obs = np.concatenate([base_obs, task_info]).astype(np.float32)
        return extended_obs
    
    def _get_task_info(self) -> Dict[str, Any]:
        """Get task-related information."""
        queue_status = self.orchestrator.get_queue_status()
        perf_metrics = self.orchestrator.get_performance_metrics()
        
        return {
            "pending_tasks": queue_status["pending_tasks"],
            "active_tasks": queue_status["active_tasks"],
            "completed_tasks": queue_status["completed_tasks"],
            "failed_tasks": queue_status["failed_tasks"],
            "success_rate": perf_metrics["success_rate"],
            "current_tasks": list(self.current_tasks.keys()),
        }
    
    def close(self):
        """Close environment."""
        super().close()
        
        # Print final statistics
        self.orchestrator.print_status()
