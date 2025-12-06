"""
PPO Neural Network Architecture
Actor-Critic model optimized for PvZ state space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class PvZActorCritic(nn.Module):
    """
    Actor-Critic network for PvZ PPO agent.
    
    Architecture:
    - Separate encoders for plant grid and zombie grid (conv layers)
    - MLP for scalar features (sun, wave, lawn mowers)
    - Shared feature fusion
    - Actor head (policy) and Critic head (value function)
    """
    
    def __init__(
        self,
        num_plant_types: int = 15,
        num_actions: int = 379,  # 7 plants * 54 positions + 1 no-op
        hidden_dim: int = 256,
    ):
        """
        Initialize actor-critic network.
        
        Args:
            num_plant_types: Number of plant types (for embedding)
            num_actions: Total number of actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.num_actions = num_actions
        
        # Plant grid encoder (6x9 grid with plant types)
        # Use embedding + conv to capture spatial patterns
        self.plant_embed = nn.Embedding(num_plant_types, 16)
        self.plant_conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.plant_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # After conv: 64 channels * 6 * 9 = 3456 -> flatten
        
        # Zombie grid encoder (6x9 grid with HP values)
        self.zombie_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.zombie_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # After conv: 32 channels * 6 * 9 = 1728 -> flatten
        
        # Scalar features encoder (sun, wave, lawn mowers)
        # sun (1) + wave (1) + lawn_mowers (6) = 8 features
        self.scalar_fc = nn.Linear(8, 64)
        
        # Feature fusion
        # 3456 (plant) + 1728 (zombie) + 64 (scalar) = 5248
        self.fusion_fc1 = nn.Linear(3456 + 1728 + 64, hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (policy)
        self.actor_fc = nn.Linear(hidden_dim, num_actions)
        
        # Critic head (value function)
        self.critic_fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.
        
        Args:
            obs: Dictionary of observation tensors
                - plant_grid: (batch, 6, 9) int32
                - zombie_grid: (batch, 6, 9) float32
                - sun: (batch, 1) float32
                - wave: (batch, 1) float32
                - lawn_mowers: (batch, 6) int32
                
        Returns:
            action_logits: (batch, num_actions)
            value: (batch, 1)
        """
        batch_size = obs["plant_grid"].shape[0]
        
        # Encode plant grid
        plant_grid = obs["plant_grid"].long()  # (batch, 6, 9)
        plant_emb = self.plant_embed(plant_grid)  # (batch, 6, 9, 16)
        plant_emb = plant_emb.permute(0, 3, 1, 2)  # (batch, 16, 6, 9)
        plant_feat = F.relu(self.plant_conv1(plant_emb))  # (batch, 32, 6, 9)
        plant_feat = F.relu(self.plant_conv2(plant_feat))  # (batch, 64, 6, 9)
        plant_feat = plant_feat.reshape(batch_size, -1)  # (batch, 3456)
        
        # Encode zombie grid
        zombie_grid = obs["zombie_grid"].unsqueeze(1)  # (batch, 1, 6, 9)
        zombie_feat = F.relu(self.zombie_conv1(zombie_grid))  # (batch, 16, 6, 9)
        zombie_feat = F.relu(self.zombie_conv2(zombie_feat))  # (batch, 32, 6, 9)
        zombie_feat = zombie_feat.reshape(batch_size, -1)  # (batch, 1728)
        
        # Encode scalar features
        scalar_feat = torch.cat([
            obs["sun"],  # (batch, 1)
            obs["wave"],  # (batch, 1)
            obs["lawn_mowers"].float(),  # (batch, 6)
        ], dim=1)  # (batch, 8)
        scalar_feat = F.relu(self.scalar_fc(scalar_feat))  # (batch, 64)
        
        # Fuse features
        fused = torch.cat([plant_feat, zombie_feat, scalar_feat], dim=1)  # (batch, 5248)
        x = F.relu(self.fusion_fc1(fused))  # (batch, hidden_dim)
        x = F.relu(self.fusion_fc2(x))  # (batch, hidden_dim)
        
        # Actor and Critic heads
        action_logits = self.actor_fc(x)  # (batch, num_actions)
        value = self.critic_fc(x)  # (batch, 1)
        
        return action_logits, value
    
    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.
        
        Args:
            obs: Observation dictionary
            action: Optional action to evaluate (for training)
            deterministic: If True, select argmax action
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            entropy: Entropy of policy
            value: State value estimate
        """
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


class PPOMemory:
    """
    Memory buffer for storing trajectories during PPO training.
    """
    
    def __init__(self, num_envs: int, num_steps: int):
        """
        Initialize memory buffer.
        
        Args:
            num_envs: Number of parallel environments
            num_steps: Number of steps to store per environment
        """
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.ptr = 0
        
        # Buffers (will be initialized on first insert)
        self.obs_buffers = {}
        self.actions = None
        self.log_probs = None
        self.values = None
        self.rewards = None
        self.dones = None
        self.advantages = None
        self.returns = None
        
    def insert(
        self,
        obs: Dict[str, np.ndarray],
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ):
        """Insert transition into buffer."""
        # Initialize buffers on first insert
        if self.actions is None:
            self._init_buffers(obs)
        
        # Store data
        for key, value in obs.items():
            self.obs_buffers[key][self.ptr] = value
        
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = values
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        
        self.ptr = (self.ptr + 1) % self.num_steps
    
    def _init_buffers(self, obs: Dict[str, np.ndarray]):
        """Initialize storage buffers based on observation shapes."""
        for key, value in obs.items():
            self.obs_buffers[key] = np.zeros(
                (self.num_steps, self.num_envs, *value.shape[1:]),
                dtype=value.dtype
            )
        
        self.actions = np.zeros((self.num_steps, self.num_envs), dtype=np.int64)
        self.log_probs = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.rewards = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.advantages = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
    
    def compute_returns_and_advantages(
        self, last_value: np.ndarray, gamma: float = 0.99, gae_lambda: float = 0.95
    ):
        """
        Compute returns and advantages using GAE.
        
        Args:
            last_value: Value estimate for last state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        last_gae = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        self.returns = self.advantages + self.values
    
    def get_batches(self, batch_size: int):
        """
        Get random batches for training.
        
        Args:
            batch_size: Size of each batch
            
        Yields:
            Dictionary of batched data
        """
        # Flatten to (num_steps * num_envs)
        total_samples = self.num_steps * self.num_envs
        indices = np.random.permutation(total_samples)
        
        for start_idx in range(0, total_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            batch = {
                "obs": {
                    key: torch.from_numpy(value.reshape(-1, *value.shape[2:])[batch_indices])
                    for key, value in self.obs_buffers.items()
                },
                "actions": torch.from_numpy(self.actions.reshape(-1)[batch_indices]),
                "log_probs": torch.from_numpy(self.log_probs.reshape(-1)[batch_indices]),
                "values": torch.from_numpy(self.values.reshape(-1)[batch_indices]),
                "advantages": torch.from_numpy(self.advantages.reshape(-1)[batch_indices]),
                "returns": torch.from_numpy(self.returns.reshape(-1)[batch_indices]),
            }
            
            yield batch
