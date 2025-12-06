"""
PPO Training Script for PvZ RL Agent
High-performance training implementation targeting 200k+ steps/sec.
"""

import os
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from vec_env import VecPvZEnv
from model import PvZActorCritic, PPOMemory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PvZ PPO agent")
    
    # Training hyperparameters
    parser.add_argument("--num-envs", type=int, default=128,
                        help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=256,
                        help="Number of steps per environment per update")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
                        help="Total timesteps to train for")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda parameter")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="PPO clipping coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value function loss coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy bonus coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="Number of minibatches per update")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="Number of epochs per PPO update")
    
    # Environment settings
    parser.add_argument("--target-waves", type=int, default=20,
                        help="Target waves to survive")
    parser.add_argument("--max-episode-steps", type=int, default=100000,
                        help="Maximum steps per episode")
    
    # Logging and checkpointing
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N updates")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Save checkpoint every N updates")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    
    # Other settings
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda)")
    
    return parser.parse_args()


class PPOTrainer:
    """PPO trainer for PvZ agent."""
    
    def __init__(self, args):
        """Initialize trainer with arguments."""
        self.args = args
        
        # Set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # Create environment
        print(f"Creating {args.num_envs} parallel environments...")
        self.envs = VecPvZEnv(
            num_envs=args.num_envs,
            seed=args.seed,
            target_waves=args.target_waves,
            max_steps=args.max_episode_steps,
        )
        
        # Create model
        print("Initializing model...")
        self.model = PvZActorCritic(num_actions=379).to(args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        
        # Create memory buffer
        self.memory = PPOMemory(args.num_envs, args.num_steps)
        
        # Training state
        self.global_step = 0
        self.num_updates = 0
        self.start_time = time.time()
        self.current_obs = None
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_waves = []
        
        # Checkpoint directory
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Resume from checkpoint if specified
        if args.resume:
            self.load_checkpoint(args.resume)
    
    def to_torch(self, obs: dict) -> dict:
        """Convert numpy observation to torch tensors."""
        return {
            key: torch.from_numpy(value).to(self.args.device)
            for key, value in obs.items()
        }
    
    def collect_rollouts(self):
        """Collect rollouts from environments."""
        obs, _ = self.envs.reset() if self.global_step == 0 else (self.current_obs, None)
        
        for step in range(self.args.num_steps):
            self.global_step += self.args.num_envs
            
            # Get action from policy
            with torch.no_grad():
                obs_torch = self.to_torch(obs)
                action, log_prob, _, value = self.model.get_action_and_value(obs_torch)
                action = action.cpu().numpy()
                log_prob = log_prob.cpu().numpy()
                value = value.cpu().numpy().squeeze()
            
            # Step environments
            next_obs, rewards, terminated, truncated, infos = self.envs.step(action)
            dones = terminated | truncated
            
            # Store transition
            self.memory.insert(obs, action, log_prob, value, rewards, dones)
            
            # Track episode statistics
            for i, (term, trunc, info) in enumerate(zip(terminated, truncated, infos)):
                if term or trunc:
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])
                        self.episode_lengths.append(info["episode"]["l"])
                    self.episode_waves.append(info.get("wave", 0))
            
            obs = next_obs
        
        # Store current observation for next rollout
        self.current_obs = obs
        
        # Compute advantages
        with torch.no_grad():
            obs_torch = self.to_torch(obs)
            _, _, _, last_value = self.model.get_action_and_value(obs_torch)
            last_value = last_value.cpu().numpy().squeeze()
        
        self.memory.compute_returns_and_advantages(
            last_value, self.args.gamma, self.args.gae_lambda
        )
    
    def update_policy(self):
        """Update policy using PPO."""
        batch_size = (self.args.num_envs * self.args.num_steps) // self.args.num_minibatches
        
        # Training metrics
        pg_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []
        
        for epoch in range(self.args.update_epochs):
            for batch in self.memory.get_batches(batch_size):
                # Move to device
                obs = {k: v.to(self.args.device) for k, v in batch["obs"].items()}
                actions = batch["actions"].to(self.args.device)
                old_log_probs = batch["log_probs"].to(self.args.device)
                advantages = batch["advantages"].to(self.args.device)
                returns = batch["returns"].to(self.args.device)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Get current policy outputs
                _, log_probs, entropy, values = self.model.get_action_and_value(obs, actions)
                values = values.squeeze()
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef) * advantages
                pg_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * ((values - returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss + self.args.vf_coef * value_loss - self.args.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                pg_losses.append(pg_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # Compute approximate KL divergence and clip fraction
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    approx_kls.append(approx_kl.item())
                    clip_fraction = ((ratio - 1).abs() > self.args.clip_coef).float().mean()
                    clip_fractions.append(clip_fraction.item())
        
        return {
            "pg_loss": np.mean(pg_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "approx_kl": np.mean(approx_kls),
            "clip_fraction": np.mean(clip_fractions),
        }
    
    def log_metrics(self, metrics: dict):
        """Log training metrics."""
        elapsed = time.time() - self.start_time
        fps = int(self.global_step / elapsed)
        
        print(f"\n{'='*60}")
        print(f"Update: {self.num_updates}")
        print(f"Global Step: {self.global_step:,} / {self.args.total_timesteps:,}")
        print(f"FPS: {fps:,}")
        print(f"Time Elapsed: {elapsed:.1f}s")
        
        if self.episode_rewards:
            print(f"\nEpisode Stats (last {len(self.episode_rewards)} episodes):")
            print(f"  Mean Reward: {np.mean(self.episode_rewards):.2f}")
            print(f"  Mean Length: {np.mean(self.episode_lengths):.0f}")
            print(f"  Mean Waves: {np.mean(self.episode_waves):.1f}")
            print(f"  Max Waves: {np.max(self.episode_waves)}")
            self.episode_rewards.clear()
            self.episode_lengths.clear()
            self.episode_waves.clear()
        
        print(f"\nLoss Metrics:")
        print(f"  Policy Loss: {metrics['pg_loss']:.4f}")
        print(f"  Value Loss: {metrics['value_loss']:.4f}")
        print(f"  Entropy: {metrics['entropy_loss']:.4f}")
        print(f"  Approx KL: {metrics['approx_kl']:.4f}")
        print(f"  Clip Fraction: {metrics['clip_fraction']:.3f}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, filename: Optional[str] = None):
        """Save training checkpoint."""
        if filename is None:
            filename = f"checkpoint_{self.num_updates}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "num_updates": self.num_updates,
            "args": vars(self.args),
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save as "latest.pt"
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.args.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.num_updates = checkpoint["num_updates"]
        
        print(f"Resumed from step {self.global_step}, update {self.num_updates}")
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.args.total_timesteps:,} timesteps")
        print(f"Updates: {self.args.total_timesteps // (self.args.num_envs * self.args.num_steps)}")
        print(f"Target: 200k+ steps/sec\n")
        
        num_updates_total = self.args.total_timesteps // (self.args.num_envs * self.args.num_steps)
        
        while self.global_step < self.args.total_timesteps:
            # Collect rollouts
            self.collect_rollouts()
            
            # Update policy
            metrics = self.update_policy()
            
            self.num_updates += 1
            
            # Log metrics
            if self.num_updates % self.args.log_interval == 0:
                self.log_metrics(metrics)
            
            # Save checkpoint
            if self.num_updates % self.args.save_interval == 0:
                self.save_checkpoint()
        
        # Final checkpoint
        self.save_checkpoint("final.pt")
        print("\nTraining complete!")
        
        # Compute and display final statistics
        elapsed = time.time() - self.start_time
        fps = int(self.global_step / elapsed)
        print(f"\nFinal Statistics:")
        print(f"  Total Steps: {self.global_step:,}")
        print(f"  Total Time: {elapsed:.1f}s ({elapsed/3600:.2f}h)")
        print(f"  Average FPS: {fps:,}")


def main():
    """Main entry point."""
    args = parse_args()
    
    print("="*60)
    print("PvZ PPO Training")
    print("="*60)
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60)
    
    trainer = PPOTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
