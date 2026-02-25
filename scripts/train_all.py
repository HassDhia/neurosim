#!/usr/bin/env python3
"""Train PPO on all three NeuroSim environments and compare against random.

Usage::

    python scripts/train_all.py
    python scripts/train_all.py --timesteps 50000 --eval-episodes 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

# Trigger env registration
import neurosim  # noqa: F401
from neurosim.envs.wrappers import wrap_for_sb3
from stable_baselines3 import PPO

ENV_IDS = [
    "neurosim/DecoderAdapt-v0",
    "neurosim/CursorControl-v0",
    "neurosim/SpellerNav-v0",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"


def evaluate_ppo(
    model: PPO,
    env: gym.Env,
    n_episodes: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate a trained PPO model over n_episodes.

    Uses deterministic prediction and resets the env with sequential seeds
    for reproducibility.
    """
    rewards: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
        rewards.append(total_reward)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }


def evaluate_random(
    env: gym.Env,
    n_episodes: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate a random agent (samples from wrapped action space)."""
    rewards: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
        rewards.append(total_reward)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }


def env_short_name(env_id: str) -> str:
    """Extract short name from env ID, e.g. 'DecoderAdapt-v0'."""
    return env_id.split("/")[-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO on all NeuroSim envs and compare vs random."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20_000,
        help="Training timesteps per environment (default: 20000).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Evaluation episodes per agent (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    # Ensure output directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for env_id in ENV_IDS:
        name = env_short_name(env_id)
        print(f"\n{'='*60}")
        print(f"  {env_id}")
        print(f"{'='*60}")

        # --- Create and wrap environment ---
        env = gym.make(env_id)
        env = wrap_for_sb3(env)

        # --- Train PPO ---
        print(f"\n  Training PPO for {args.timesteps:,} timesteps...")
        t0 = time.time()
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            seed=args.seed,
            n_steps=512,
            batch_size=64,
            n_epochs=5,
            learning_rate=3e-4,
        )
        model.learn(total_timesteps=args.timesteps)
        train_time = time.time() - t0
        print(f"  Training completed in {train_time:.1f}s")

        # --- Save model ---
        save_path = MODELS_DIR / name
        model.save(str(save_path))
        print(f"  Model saved to {save_path}.zip")

        # --- Evaluate PPO ---
        print(f"\n  Evaluating PPO ({args.eval_episodes} episodes)...")
        ppo_results = evaluate_ppo(model, env, args.eval_episodes, args.seed)
        print(f"  PPO mean reward: {ppo_results['mean_reward']:.4f} "
              f"(+/- {ppo_results['std_reward']:.4f})")

        # --- Evaluate Random ---
        print(f"  Evaluating Random ({args.eval_episodes} episodes)...")
        random_results = evaluate_random(env, args.eval_episodes, args.seed)
        print(f"  Random mean reward: {random_results['mean_reward']:.4f} "
              f"(+/- {random_results['std_reward']:.4f})")

        improvement = ppo_results["mean_reward"] - random_results["mean_reward"]

        all_results[env_id] = {
            "env_id": env_id,
            "timesteps": args.timesteps,
            "train_time_s": round(train_time, 2),
            "ppo_mean_reward": ppo_results["mean_reward"],
            "ppo_std_reward": ppo_results["std_reward"],
            "random_mean_reward": random_results["mean_reward"],
            "random_std_reward": random_results["std_reward"],
            "improvement": improvement,
        }

        env.close()

    # --- Print comparison table ---
    print(f"\n\n{'='*80}")
    print("  COMPARISON TABLE")
    print(f"{'='*80}")
    header = f"{'Environment':<30} {'Random Mean':>12} {'PPO Mean':>12} {'Improvement':>12}"
    print(header)
    print("-" * len(header))
    for env_id, r in all_results.items():
        name = env_short_name(env_id)
        print(
            f"{name:<30} "
            f"{r['random_mean_reward']:>12.4f} "
            f"{r['ppo_mean_reward']:>12.4f} "
            f"{r['improvement']:>12.4f}"
        )
    print()

    # --- Save results JSON ---
    results_path = RESULTS_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "timesteps": args.timesteps,
                "eval_episodes": args.eval_episodes,
                "environments": all_results,
            },
            f,
            indent=2,
        )
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
