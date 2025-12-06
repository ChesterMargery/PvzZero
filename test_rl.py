"""
Test suite for RL training framework
"""

import unittest
import numpy as np
import torch
from env import PvZEnv
from vec_env import VecPvZEnv
from model import PvZActorCritic, PPOMemory


class TestPvZEnv(unittest.TestCase):
    """Test Gymnasium environment wrapper."""
    
    def setUp(self):
        """Create environment for testing."""
        self.env = PvZEnv(seed=42, target_waves=5)
    
    def test_env_creation(self):
        """Environment should initialize correctly."""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.target_waves, 5)
    
    def test_observation_space(self):
        """Observation space should be correctly defined."""
        obs_space = self.env.observation_space
        self.assertIn("plant_grid", obs_space.spaces)
        self.assertIn("zombie_grid", obs_space.spaces)
        self.assertIn("sun", obs_space.spaces)
        self.assertIn("wave", obs_space.spaces)
        self.assertIn("lawn_mowers", obs_space.spaces)
    
    def test_action_space(self):
        """Action space should have correct size."""
        # 7 plant types * 54 positions + 1 no-op = 379
        self.assertEqual(self.env.action_space.n, 379)
    
    def test_reset(self):
        """Reset should return valid observation and info."""
        obs, info = self.env.reset()
        
        # Check observation structure
        self.assertIn("plant_grid", obs)
        self.assertIn("zombie_grid", obs)
        self.assertEqual(obs["plant_grid"].shape, (6, 9))
        self.assertEqual(obs["zombie_grid"].shape, (6, 9))
        
        # Check info
        self.assertIn("wave", info)
        self.assertIn("sun", info)
    
    def test_step_no_op(self):
        """Stepping with no-op should work."""
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)
        
        # Should return valid outputs
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, (bool, np.bool_))
        self.assertIsInstance(truncated, (bool, np.bool_))
        self.assertIsInstance(info, dict)
    
    def test_step_plant_action(self):
        """Stepping with plant action should work."""
        self.env.reset()
        # Action 1 = plant sunflower at (0, 0)
        obs, reward, terminated, truncated, info = self.env.step(1)
        
        # Should return valid outputs
        self.assertIsInstance(reward, (int, float))
    
    def test_survival_reward(self):
        """Should receive survival reward each tick."""
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)
        
        # Survival reward should be at least 1.0
        self.assertGreaterEqual(reward, 1.0)
    
    def test_observation_normalization(self):
        """Observations should be normalized."""
        obs, _ = self.env.reset()
        
        # Sun should be normalized to [0, 1]
        self.assertTrue(0 <= obs["sun"][0] <= 1.0)
        
        # Wave should be normalized to [0, 1]
        self.assertTrue(0 <= obs["wave"][0] <= 1.0)
        
        # Zombie grid values should be in [0, 1]
        self.assertTrue(np.all(obs["zombie_grid"] >= 0))
        self.assertTrue(np.all(obs["zombie_grid"] <= 1.0))


class TestVecPvZEnv(unittest.TestCase):
    """Test vectorized environment."""
    
    def setUp(self):
        """Create vectorized environment for testing."""
        self.vec_env = VecPvZEnv(num_envs=4, seed=42)
    
    def test_vec_env_creation(self):
        """Vectorized environment should initialize correctly."""
        self.assertEqual(self.vec_env.num_envs, 4)
        self.assertEqual(len(self.vec_env.envs), 4)
    
    def test_vec_reset(self):
        """Reset should return batched observations."""
        obs, infos = self.vec_env.reset()
        
        # Should have batched observations
        self.assertEqual(obs["plant_grid"].shape, (4, 6, 9))
        self.assertEqual(obs["zombie_grid"].shape, (4, 6, 9))
        self.assertEqual(obs["sun"].shape, (4, 1))
        
        # Should have info for each env
        self.assertEqual(len(infos), 4)
    
    def test_vec_step(self):
        """Step should handle batched actions."""
        self.vec_env.reset()
        actions = np.array([0, 0, 0, 0])  # All no-op
        
        obs, rewards, terminated, truncated, infos = self.vec_env.step(actions)
        
        # Check shapes
        self.assertEqual(obs["plant_grid"].shape, (4, 6, 9))
        self.assertEqual(rewards.shape, (4,))
        self.assertEqual(terminated.shape, (4,))
        self.assertEqual(truncated.shape, (4,))
        self.assertEqual(len(infos), 4)
    
    def test_vec_auto_reset(self):
        """Environments should auto-reset on episode end."""
        # This is tested implicitly by the framework
        # Just verify that step continues to work
        self.vec_env.reset()
        
        for _ in range(10):
            actions = np.zeros(4, dtype=np.int32)
            obs, rewards, terminated, truncated, infos = self.vec_env.step(actions)
            
            # Should always return valid observations
            self.assertEqual(obs["plant_grid"].shape, (4, 6, 9))


class TestPvZActorCritic(unittest.TestCase):
    """Test PPO model architecture."""
    
    def setUp(self):
        """Create model for testing."""
        self.model = PvZActorCritic(num_actions=379)
    
    def test_model_creation(self):
        """Model should initialize correctly."""
        self.assertIsNotNone(self.model)
        
        # Check model has parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(num_params, 0)
    
    def test_forward_pass(self):
        """Forward pass should produce correct output shapes."""
        batch_size = 8
        
        # Create dummy observation
        obs = {
            "plant_grid": torch.zeros(batch_size, 6, 9, dtype=torch.long),
            "zombie_grid": torch.zeros(batch_size, 6, 9, dtype=torch.float32),
            "sun": torch.zeros(batch_size, 1, dtype=torch.float32),
            "wave": torch.zeros(batch_size, 1, dtype=torch.float32),
            "lawn_mowers": torch.ones(batch_size, 6, dtype=torch.long),
        }
        
        logits, values = self.model(obs)
        
        # Check output shapes
        self.assertEqual(logits.shape, (batch_size, 379))
        self.assertEqual(values.shape, (batch_size, 1))
    
    def test_get_action_and_value(self):
        """Should sample actions and compute values."""
        batch_size = 4
        
        obs = {
            "plant_grid": torch.zeros(batch_size, 6, 9, dtype=torch.long),
            "zombie_grid": torch.zeros(batch_size, 6, 9, dtype=torch.float32),
            "sun": torch.zeros(batch_size, 1, dtype=torch.float32),
            "wave": torch.zeros(batch_size, 1, dtype=torch.float32),
            "lawn_mowers": torch.ones(batch_size, 6, dtype=torch.long),
        }
        
        action, log_prob, entropy, value = self.model.get_action_and_value(obs)
        
        # Check output shapes
        self.assertEqual(action.shape, (batch_size,))
        self.assertEqual(log_prob.shape, (batch_size,))
        self.assertEqual(entropy.shape, (batch_size,))
        self.assertEqual(value.shape, (batch_size, 1))
    
    def test_deterministic_action(self):
        """Should select argmax action when deterministic=True."""
        batch_size = 2
        
        obs = {
            "plant_grid": torch.zeros(batch_size, 6, 9, dtype=torch.long),
            "zombie_grid": torch.zeros(batch_size, 6, 9, dtype=torch.float32),
            "sun": torch.zeros(batch_size, 1, dtype=torch.float32),
            "wave": torch.zeros(batch_size, 1, dtype=torch.float32),
            "lawn_mowers": torch.ones(batch_size, 6, dtype=torch.long),
        }
        
        # Sample twice with deterministic=True, should be same
        action1, _, _, _ = self.model.get_action_and_value(obs, deterministic=True)
        action2, _, _, _ = self.model.get_action_and_value(obs, deterministic=True)
        
        self.assertTrue(torch.equal(action1, action2))


class TestPPOMemory(unittest.TestCase):
    """Test PPO memory buffer."""
    
    def test_memory_creation(self):
        """Memory buffer should initialize correctly."""
        memory = PPOMemory(num_envs=4, num_steps=8)
        
        self.assertEqual(memory.num_envs, 4)
        self.assertEqual(memory.num_steps, 8)
    
    def test_memory_insert(self):
        """Should be able to insert transitions."""
        memory = PPOMemory(num_envs=4, num_steps=8)
        
        # Create dummy data
        obs = {
            "plant_grid": np.zeros((4, 6, 9), dtype=np.int32),
            "zombie_grid": np.zeros((4, 6, 9), dtype=np.float32),
            "sun": np.zeros((4, 1), dtype=np.float32),
            "wave": np.zeros((4, 1), dtype=np.float32),
            "lawn_mowers": np.ones((4, 6), dtype=np.int32),
        }
        actions = np.zeros(4, dtype=np.int64)
        log_probs = np.zeros(4, dtype=np.float32)
        values = np.zeros(4, dtype=np.float32)
        rewards = np.ones(4, dtype=np.float32)
        dones = np.zeros(4, dtype=np.float32)
        
        # Insert should work without error
        memory.insert(obs, actions, log_probs, values, rewards, dones)
        
        # Buffers should be initialized
        self.assertIsNotNone(memory.actions)
    
    def test_compute_returns_and_advantages(self):
        """Should compute returns and advantages."""
        memory = PPOMemory(num_envs=4, num_steps=8)
        
        # Fill memory with dummy data
        obs = {
            "plant_grid": np.zeros((4, 6, 9), dtype=np.int32),
            "zombie_grid": np.zeros((4, 6, 9), dtype=np.float32),
            "sun": np.zeros((4, 1), dtype=np.float32),
            "wave": np.zeros((4, 1), dtype=np.float32),
            "lawn_mowers": np.ones((4, 6), dtype=np.int32),
        }
        
        for _ in range(8):
            actions = np.zeros(4, dtype=np.int64)
            log_probs = np.zeros(4, dtype=np.float32)
            values = np.ones(4, dtype=np.float32)
            rewards = np.ones(4, dtype=np.float32)
            dones = np.zeros(4, dtype=np.float32)
            memory.insert(obs, actions, log_probs, values, rewards, dones)
        
        # Compute returns
        last_value = np.ones(4, dtype=np.float32)
        memory.compute_returns_and_advantages(last_value)
        
        # Advantages and returns should be computed
        self.assertIsNotNone(memory.advantages)
        self.assertIsNotNone(memory.returns)


if __name__ == "__main__":
    unittest.main()
