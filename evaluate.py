"""
Evaluate a trained PPO agent on PvZ environment.
Load a checkpoint and run episodes to see agent performance.
"""

import argparse
import torch
import numpy as np

from env import PvZEnv
from model import PvZActorCritic


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained PvZ agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--target-waves", type=int, default=20,
                        help="Target waves to survive")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic policy (argmax)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda)")
    return parser.parse_args()


def evaluate(args):
    """Run evaluation."""
    print("=" * 60)
    print("PvZ Agent Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Target Waves: {args.target_waves}")
    print(f"Deterministic: {args.deterministic}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = PvZActorCritic(num_actions=379).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Loaded checkpoint from step {checkpoint['global_step']}")
    
    # Create environment
    env = PvZEnv(seed=args.seed, target_waves=args.target_waves)
    
    # Run episodes
    episode_rewards = []
    episode_lengths = []
    episode_waves = []
    max_waves_reached = 0
    
    for episode in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed + episode if args.seed else None)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Convert observation to torch
            obs_torch = {
                key: torch.from_numpy(value).unsqueeze(0).to(args.device)
                for key, value in obs.items()
            }
            
            # Get action from policy
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(
                    obs_torch, deterministic=args.deterministic
                )
                action = action.item()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        # Record episode statistics
        final_wave = info.get("wave", 0)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_waves.append(final_wave)
        max_waves_reached = max(max_waves_reached, final_wave)
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length}")
        print(f"  Waves: {final_wave}")
        print(f"  Victory: {'Yes' if final_wave >= args.target_waves else 'No'}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Episodes: {args.num_episodes}")
    print(f"\nReward:")
    print(f"  Mean: {np.mean(episode_rewards):.2f}")
    print(f"  Std: {np.std(episode_rewards):.2f}")
    print(f"  Min: {np.min(episode_rewards):.2f}")
    print(f"  Max: {np.max(episode_rewards):.2f}")
    print(f"\nLength:")
    print(f"  Mean: {np.mean(episode_lengths):.0f}")
    print(f"  Std: {np.std(episode_lengths):.0f}")
    print(f"\nWaves:")
    print(f"  Mean: {np.mean(episode_waves):.1f}")
    print(f"  Std: {np.std(episode_waves):.1f}")
    print(f"  Max: {max_waves_reached}")
    print(f"\nSuccess Rate: {sum(1 for w in episode_waves if w >= args.target_waves) / args.num_episodes * 100:.1f}%")
    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
