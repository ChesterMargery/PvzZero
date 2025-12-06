# PvzZero

High-fidelity, high-performance headless Plants vs. Zombies Survival Endless simulator for Reinforcement Learning.

## Features

- **>10,000 FPS** simulation speed
- **100% logic parity** with original game
- **RL Training Framework** with PPO implementation
- **Vectorized environments** for parallel simulation
- **Gymnasium interface** for easy integration

## Quick Start

### Run Tests

```bash
python -m unittest tests.py
```

### Train an AI Agent

```bash
pip install -r requirements.txt
python train.py --num-envs 128 --total-timesteps 10000000
```

See [RL_TRAINING.md](RL_TRAINING.md) for detailed training documentation.

## RL Training Framework

The project includes a complete reinforcement learning training framework:

- **Gymnasium Environment** (`env.py`): Standard RL environment wrapper
- **Vectorized Environment** (`vec_env.py`): Parallel simulation for efficiency
- **PPO Model** (`model.py`): Actor-Critic neural network optimized for PvZ
- **Training Script** (`train.py`): Complete PPO implementation with logging and checkpointing

**Performance**: Achieves ~40k FPS pure simulation and ~1.5k training FPS with neural networks.

**Target**: Train AI agents that can survive 20+ waves in Survival Endless mode.

## Repository Structure

```
PvzZero/
├── consts.py           # Game constants and enums
├── state.py            # Entity dataclasses (Plant, Zombie, etc.)
├── physics.py          # Collision detection and physics
├── engine.py           # Main game engine and simulation loop
├── tests.py            # Unit tests for simulator
├── env.py              # Gymnasium environment wrapper
├── vec_env.py          # Vectorized environment
├── model.py            # PPO neural network
├── train.py            # Training script
├── test_rl.py          # RL framework tests
└── RL_TRAINING.md      # Detailed training documentation
```