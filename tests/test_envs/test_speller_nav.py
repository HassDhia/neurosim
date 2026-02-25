"""Tests for SpellerNav-v0 environment."""
import gymnasium as gym
import numpy as np
import pytest

import neurosim.envs  # noqa: F401 — triggers gymnasium env registration


class TestSpellerNavEnv:
    def test_env_creation(self):
        env = gym.make("neurosim/SpellerNav-v0")
        assert env is not None
        env.close()

    def test_reset_returns_obs_and_info(self):
        env = gym.make("neurosim/SpellerNav-v0")
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        env.close()

    def test_step_returns_correct_tuple(self):
        env = gym.make("neurosim/SpellerNav-v0")
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_observation_space_contains_obs(self):
        env = gym.make("neurosim/SpellerNav-v0")
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        env.close()

    def test_render_ansi_returns_string(self):
        env = gym.make("neurosim/SpellerNav-v0", render_mode="ansi")
        env.reset(seed=42)
        action = env.action_space.sample()
        env.step(action)
        output = env.render()
        assert isinstance(output, str)
        assert len(output) > 0
        env.close()

    def test_posterior_sums_to_one(self):
        env = gym.make("neurosim/SpellerNav-v0")
        obs, _ = env.reset(seed=42)
        posterior = obs["posterior"]
        np.testing.assert_allclose(
            posterior.sum(), 1.0, atol=1e-5,
            err_msg="Initial posterior should sum to 1.0",
        )
        env.close()

    def test_commit_advances_queue(self):
        env = gym.make("neurosim/SpellerNav-v0")
        env.reset(seed=42)
        # Step with commit=1 to select a symbol
        action = {"stimulus_group": 0, "commit": 1}
        _, _, _, _, info = env.step(action)
        assert info["queue_idx"] == 1, (
            f"Committing should advance queue_idx to 1, got {info['queue_idx']}"
        )
        env.close()

    def test_flash_count_increments(self):
        env = gym.make("neurosim/SpellerNav-v0")
        env.reset(seed=42)
        # Step with commit=0 (just flash, no commit)
        action = {"stimulus_group": 0, "commit": 0}
        _, _, _, _, info = env.step(action)
        assert info["flash_count"] > 0, (
            f"Flash count should be positive after a flash step, got {info['flash_count']}"
        )
        env.close()
