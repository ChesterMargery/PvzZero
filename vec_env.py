"""
Vectorized PvZ Environment
High-performance parallel environment for efficient RL training.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Any
import gymnasium as gym

from env import PvZEnv


class VecPvZEnv:
    """
    Vectorized PvZ environment for parallel simulation.
    
    Runs multiple independent PvZ environments in parallel to maximize
    sample efficiency. Target: 200k+ steps/second on 4-core CPU.
    """
    
    def __init__(
        self,
        num_envs: int = 64,
        seed: Optional[int] = None,
        target_waves: int = 20,
        max_steps: int = 100000,
    ):
        """
        Initialize vectorized environment.
        
        Args:
            num_envs: Number of parallel environments
            seed: Base random seed
            target_waves: Target waves to survive
            max_steps: Max steps per episode
        """
        self.num_envs = num_envs
        self.target_waves = target_waves
        self.max_steps = max_steps
        
        # Create parallel environments with different seeds
        self.envs = [
            PvZEnv(
                seed=seed + i if seed is not None else None,
                target_waves=target_waves,
                max_steps=max_steps,
            )
            for i in range(num_envs)
        ]
        
        # Use first env for space definitions
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # Track which environments need reset
        self._needs_reset = np.zeros(num_envs, dtype=bool)
        
    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        """
        Reset all environments.
        
        Args:
            seed: Base random seed
            
        Returns:
            Batched observations and info dicts
        """
        obs_list = []
        info_list = []
        
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            info_list.append(info)
        
        self._needs_reset[:] = False
        
        # Stack observations
        batched_obs = self._stack_obs(obs_list)
        
        return batched_obs, info_list
    
    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all environments with given actions.
        
        Args:
            actions: Array of actions for each environment (num_envs,)
            
        Returns:
            observations, rewards, terminated, truncated, infos
        """
        obs_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        info_list = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            # Auto-reset if needed
            if self._needs_reset[i]:
                obs, info = env.reset()
                obs_list.append(obs)
                info_list.append(info)
                # On reset, no reward/done signals for this step
                rewards[i] = 0.0
                terminated[i] = False
                truncated[i] = False
                self._needs_reset[i] = False
            else:
                obs, reward, term, trunc, info = env.step(action)
                obs_list.append(obs)
                rewards[i] = reward
                terminated[i] = term
                truncated[i] = trunc
                info_list.append(info)
                
                # Mark for reset if episode ended
                if term or trunc:
                    self._needs_reset[i] = True
        
        # Stack observations
        batched_obs = self._stack_obs(obs_list)
        
        return batched_obs, rewards, terminated, truncated, info_list
    
    def _stack_obs(self, obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Stack list of observations into batched observation.
        
        Args:
            obs_list: List of observation dicts
            
        Returns:
            Batched observation dict
        """
        # Stack each key separately
        batched_obs = {}
        for key in obs_list[0].keys():
            batched_obs[key] = np.stack([obs[key] for obs in obs_list], axis=0)
        return batched_obs
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()
    
    @property
    def num_steps_per_env(self) -> np.ndarray:
        """Get current step count for each environment."""
        return np.array([env._episode_steps for env in self.envs])


# Note: For true multiprocessing support, consider using stable_baselines3.SubprocVecEnv
# For PvZSim, single-process VecPvZEnv is often faster due to the simulator being
# pure Python and the GIL. Multiprocessing overhead typically reduces performance
# compared to the vectorized single-process implementation above.
