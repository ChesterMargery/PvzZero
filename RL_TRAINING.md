# PvZ Zero Reinforcement Learning Training

This document describes how to use the reinforcement learning training framework for PvZ Zero.

## Overview

The RL training framework enables training AI agents to play Plants vs. Zombies Survival Endless mode using Proximal Policy Optimization (PPO). The framework achieves:

- **40,000+ FPS** pure simulation speed (64 environments)
- **1,500+ FPS** training speed (with neural network inference)
- Support for **64-256 parallel environments**
- Automatic checkpointing and resume functionality

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch>=2.6.0` - PyTorch for neural networks
- `numpy>=1.24.0` - Numerical operations
- `gymnasium>=0.29.0` - RL environment interface

### 2. Run Training

Basic training with default settings:

```bash
python train.py
```

Custom training configuration:

```bash
python train.py \
  --num-envs 128 \
  --total-timesteps 10000000 \
  --target-waves 20 \
  --learning-rate 0.0003 \
  --log-interval 10 \
  --save-interval 100
```

### 3. Resume Training

To resume from a checkpoint:

```bash
python train.py --resume checkpoints/checkpoint_100.pt
```

## Command Line Arguments

### Environment Settings
- `--num-envs`: Number of parallel environments (default: 128)
- `--target-waves`: Target waves to survive (default: 20)
- `--max-episode-steps`: Maximum steps per episode (default: 100000)

### Training Hyperparameters
- `--num-steps`: Steps per environment per update (default: 256)
- `--total-timesteps`: Total training timesteps (default: 10,000,000)
- `--learning-rate`: Learning rate (default: 0.0003)
- `--gamma`: Discount factor (default: 0.99)
- `--gae-lambda`: GAE lambda parameter (default: 0.95)
- `--clip-coef`: PPO clipping coefficient (default: 0.2)
- `--vf-coef`: Value function loss coefficient (default: 0.5)
- `--ent-coef`: Entropy bonus coefficient (default: 0.01)
- `--max-grad-norm`: Maximum gradient norm (default: 0.5)
- `--num-minibatches`: Number of minibatches per update (default: 4)
- `--update-epochs`: Number of epochs per PPO update (default: 4)

### Logging and Checkpointing
- `--log-interval`: Log every N updates (default: 10)
- `--save-interval`: Save checkpoint every N updates (default: 100)
- `--checkpoint-dir`: Directory to save checkpoints (default: "checkpoints")
- `--resume`: Path to checkpoint to resume from

### Other Settings
- `--seed`: Random seed (default: 42)
- `--device`: Device to use: "cpu" or "cuda" (default: "cpu")

## Environment Details

### Observation Space

The agent observes:

1. **Plant Grid** (6×9): Grid showing which plants are placed where
2. **Zombie Grid** (6×9): Grid showing zombie positions and HP (normalized)
3. **Sun**: Current sun amount (normalized to [0, 1])
4. **Wave**: Current wave number (normalized to [0, 1])
5. **Lawn Mowers**: Binary array indicating which lawn mowers are still active

### Action Space

The agent can choose from 379 discrete actions:

- **Action 0**: No-op (do nothing)
- **Actions 1-378**: Plant a specific plant type at a specific position
  - 7 plant types: Sunflower, Fume-shroom, Gloom-shroom, Winter Melon, Pumpkin, Lily Pad, Cherry Bomb
  - 54 positions: 6 rows × 9 columns

### Reward Function

The agent receives rewards based on:

- **+1** for each tick survived
- **+50** for each zombie killed
- **+0.1** for each sun collected
- **+100** bonus for completing a wave
- **-1000** penalty for game over
- **+5000** bonus for reaching target waves

## Model Architecture

The PPO agent uses an Actor-Critic architecture with:

- **Plant Grid Encoder**: Embedding + 2-layer CNN
- **Zombie Grid Encoder**: 2-layer CNN
- **Scalar Feature Encoder**: MLP for sun/wave/lawn mowers
- **Feature Fusion**: Combines all encoders
- **Actor Head**: Outputs action logits
- **Critic Head**: Outputs state value estimate

Total parameters: ~1.5 million

## Training on GitHub Actions

You can trigger training runs on GitHub Actions:

1. Go to the "Actions" tab in the repository
2. Select "Train PvZ RL Agent" workflow
3. Click "Run workflow"
4. Configure parameters (optional):
   - Number of parallel environments
   - Total timesteps
   - Target waves
5. Click "Run workflow"

Trained models will be saved as artifacts and can be downloaded after training completes.

## Performance Tips

### For Faster Training

1. **Increase parallel environments**: Use `--num-envs 256` for more samples per update
2. **Adjust update frequency**: Use `--num-steps 512` to collect more data before each update
3. **Use GPU**: Add `--device cuda` if you have a CUDA-capable GPU (requires CUDA-enabled PyTorch)

### For Better Agent Performance

1. **Tune rewards**: Modify reward function in `env.py` to encourage desired behaviors
2. **Adjust learning rate**: Try `--learning-rate 0.0001` for more stable learning
3. **Train longer**: Use `--total-timesteps 50000000` for extended training
4. **Experiment with plants**: Modify `plant_types` list in `env.py` to try different plant combinations

## Testing

Run the test suite to verify the framework:

```bash
# Test RL framework
python -m unittest test_rl.py -v

# Test original simulator
python -m unittest tests.py -v
```

All tests should pass.

## File Structure

```
PvzZero/
├── env.py              # Gymnasium environment wrapper
├── vec_env.py          # Vectorized environment for parallel simulation
├── model.py            # PPO neural network architecture
├── train.py            # Main training script
├── test_rl.py          # RL framework tests
├── requirements.txt    # Python dependencies
├── checkpoints/        # Training checkpoints (created during training)
└── .github/
    └── workflows/
        └── train.yml   # GitHub Actions training workflow
```

## Troubleshooting

### Low FPS

- **Problem**: Training FPS is much lower than expected
- **Solution**: 
  - Reduce `--num-envs` if you're running out of memory
  - Increase `--num-steps` to amortize NN overhead
  - Use GPU with `--device cuda`

### Agent Not Learning

- **Problem**: Agent performance doesn't improve
- **Solution**:
  - Check reward function is appropriate
  - Try lower learning rate: `--learning-rate 0.0001`
  - Increase training time: `--total-timesteps 50000000`
  - Adjust PPO clipping: `--clip-coef 0.1`

### Out of Memory

- **Problem**: Training crashes with OOM error
- **Solution**:
  - Reduce `--num-envs` to use fewer parallel environments
  - Reduce `--num-steps` to use less memory per update
  - Use smaller batches: `--num-minibatches 8`

## Next Steps

After training, you can:

1. **Evaluate the agent**: Load the checkpoint and test on different scenarios
2. **Fine-tune**: Resume training with adjusted hyperparameters
3. **Experiment**: Try different plant combinations and reward functions
4. **Scale up**: Train for longer with more environments

For questions or issues, please open an issue on the GitHub repository.
