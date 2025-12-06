"""
PvZ Gymnasium Environment
Wraps PvZSim as a standard Gymnasium environment for reinforcement learning.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from engine import PvZSim
from consts import PlantType, NUM_ROWS, NUM_COLS


class PvZEnv(gym.Env):
    """
    Plants vs. Zombies Gymnasium Environment.
    
    Observation Space:
        - Plant grid (6x9): one-hot encoded plant types
        - Zombie positions: grid with zombie counts and HP
        - Sun amount: normalized
        - Wave number: normalized
        - Lawn mower status: binary array
    
    Action Space:
        - Discrete actions: plant_type * positions + no-op
        - Actions: 0 = no-op, 1-126 = plant specific type at (row, col)
    
    Reward Function:
        - +1 for each tick survived
        - +50 for each zombie killed
        - +10 for collecting sun
        - -1000 for game over (death)
        - +5000 for winning (completing target waves)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        seed: Optional[int] = None,
        target_waves: int = 20,
        max_steps: int = 100000,
    ):
        """
        Initialize PvZ environment.
        
        Args:
            seed: Random seed for reproducibility
            target_waves: Number of waves to survive for victory
            max_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.sim = PvZSim(seed=seed)
        self.target_waves = target_waves
        self.max_steps = max_steps
        self._episode_steps = 0
        
        # Track previous state for reward calculation
        self._prev_zombies_killed = 0
        self._prev_sun = 50
        
        # Define observation space
        # Plant grid: 6x9 with plant type encoding (14 plant types + empty)
        # Zombie grid: 6x9 with zombie count/hp aggregated
        # Sun: single normalized value
        # Wave: single normalized value
        # Lawn mowers: 6 binary values
        self.observation_space = spaces.Dict({
            "plant_grid": spaces.Box(
                low=0, high=len(PlantType), shape=(NUM_ROWS, NUM_COLS), dtype=np.int32
            ),
            "zombie_grid": spaces.Box(
                low=0, high=1.0, shape=(NUM_ROWS, NUM_COLS), dtype=np.float32
            ),
            "sun": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "wave": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "lawn_mowers": spaces.Box(low=0, high=1, shape=(NUM_ROWS,), dtype=np.int32),
        })
        
        # Define action space
        # 0 = no-op
        # 1-54 = PlantType.SUNFLOWER at (0,0) to (5,8)
        # 55-108 = PlantType.FUME_SHROOM at (0,0) to (5,8)
        # 109-162 = PlantType.WINTER_MELON at (0,0) to (5,8)
        # etc.
        # We'll use a subset of useful plants for simplicity
        self.plant_types = [
            PlantType.SUNFLOWER,       # Economy
            PlantType.FUME_SHROOM,     # Main damage dealer
            PlantType.GLOOM_SHROOM,    # AOE damage
            PlantType.WINTER_MELON,    # Slow + damage
            PlantType.PUMPKIN,         # Defense
            PlantType.LILY_PAD,        # Water platform
            PlantType.CHERRY_BOMB,     # Instant clear
        ]
        
        # Action: 0 for no-op, then plant_type_idx * 54 + row * 9 + col + 1
        num_plant_actions = len(self.plant_types) * NUM_ROWS * NUM_COLS
        self.action_space = spaces.Discrete(num_plant_actions + 1)
        
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Get current observation from simulator state.
        
        Returns:
            Dictionary containing observation arrays
        """
        # Plant grid: encode plant types
        plant_grid = np.zeros((NUM_ROWS, NUM_COLS), dtype=np.int32)
        for plant in self.sim.plants:
            if plant.is_alive():
                plant_grid[plant.mRow, plant.mCol] = int(plant.mPlantType) + 1
        
        # Zombie grid: aggregate zombie HP per cell
        zombie_grid = np.zeros((NUM_ROWS, NUM_COLS), dtype=np.float32)
        for zombie in self.sim.zombies:
            if zombie.is_alive():
                # Convert zombie x position to column
                col = int((zombie.mX - 40) / 80)  # GRID_OFFSET_X=40, CELL_WIDTH=80
                col = max(0, min(NUM_COLS - 1, col))
                row = zombie.mRow
                # Normalize HP (max ~6000 for Giga-Garg)
                zombie_grid[row, col] += min(zombie.mHP / 6000.0, 1.0)
        
        # Normalize sun (cap at 9990 which is max)
        sun = np.array([min(self.sim.sun / 10000.0, 1.0)], dtype=np.float32)
        
        # Normalize wave (assume max 100 waves)
        wave = np.array([min(self.sim.wave / 100.0, 1.0)], dtype=np.float32)
        
        # Lawn mowers status
        lawn_mowers = np.array(
            [1 if mower.mIsActive else 0 for mower in self.sim.lawn_mowers],
            dtype=np.int32
        )
        
        return {
            "plant_grid": plant_grid,
            "zombie_grid": zombie_grid,
            "sun": sun,
            "wave": wave,
            "lawn_mowers": lawn_mowers,
        }
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on current state.
        
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Survival reward: +1 per tick
        reward += 1.0
        
        # Zombie kill reward
        current_zombies = len([z for z in self.sim.zombies if z.is_alive()])
        zombies_killed = self._prev_zombies_killed - current_zombies
        if zombies_killed > 0:
            reward += 50.0 * zombies_killed
        self._prev_zombies_killed = current_zombies
        
        # Sun collection reward (encourage economy)
        sun_gained = self.sim.sun - self._prev_sun
        if sun_gained > 0:
            reward += 0.1 * sun_gained  # Small reward for sun collection
        self._prev_sun = self.sim.sun
        
        # Wave completion bonus
        if self.sim.wave > 0 and len([z for z in self.sim.zombies if z.is_alive()]) == 0:
            reward += 100.0
        
        # Game over penalty
        if self.sim.game_over:
            reward -= 1000.0
        
        # Victory bonus
        if self.sim.wave >= self.target_waves:
            reward += 5000.0
        
        return reward
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        self.sim.reset(seed=seed)
        self._episode_steps = 0
        self._prev_zombies_killed = 0
        self._prev_sun = 50
        
        obs = self._get_obs()
        info = {
            "wave": self.sim.wave,
            "sun": self.sim.sun,
            "zombies": len([z for z in self.sim.zombies if z.is_alive()]),
        }
        
        return obs, info
    
    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0=no-op, 1+=plant action)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self._episode_steps += 1
        
        # Decode and execute action
        if action > 0:
            action_idx = action - 1
            plant_type_idx = action_idx // (NUM_ROWS * NUM_COLS)
            pos_idx = action_idx % (NUM_ROWS * NUM_COLS)
            row = pos_idx // NUM_COLS
            col = pos_idx % NUM_COLS
            
            if plant_type_idx < len(self.plant_types):
                plant_type = self.plant_types[plant_type_idx]
                # Try to plant (may fail due to cost or invalid placement)
                self.sim.plant_at(plant_type, row, col, free=False)
        
        # Step simulator (one tick)
        self.sim.step()
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self.sim.game_over or self.sim.wave >= self.target_waves
        truncated = self._episode_steps >= self.max_steps
        
        # Info dict
        info = {
            "wave": self.sim.wave,
            "sun": self.sim.sun,
            "zombies": len([z for z in self.sim.zombies if z.is_alive()]),
            "plants": len([p for p in self.sim.plants if p.is_alive()]),
            "episode_steps": self._episode_steps,
        }
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Clean up environment resources."""
        pass
