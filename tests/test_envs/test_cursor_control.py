"""Tests for CursorControl-v0 environment."""
import gymnasium as gym
import numpy as np
import pytest

import neurosim.envs  # noqa: F401 — triggers gymnasium env registration


class TestCursorControlEnv:
    def test_env_creation(self):
        env = gym.make("neurosim/CursorControl-v0")
        assert env is not None
        env.close()

    def test_reset_returns_obs_and_info(self):
        env = gym.make("neurosim/CursorControl-v0")
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        env.close()

    def test_step_returns_correct_tuple(self):
        env = gym.make("neurosim/CursorControl-v0")
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
        env = gym.make("neurosim/CursorControl-v0")
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        env.close()

    def test_render_ansi_returns_string(self):
        env = gym.make("neurosim/CursorControl-v0", render_mode="ansi")
        env.reset(seed=42)
        action = env.action_space.sample()
        env.step(action)
        output = env.render()
        assert isinstance(output, str)
        assert len(output) > 0
        env.close()

    def test_info_contains_distance_key(self):
        env = gym.make("neurosim/CursorControl-v0")
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, info = env.step(action)
        assert "distance_to_target" in obs, (
            "observation should contain 'distance_to_target'"
        )
        dist = float(obs["distance_to_target"][0])
        assert np.isfinite(dist)
        assert dist >= 0.0
        env.close()

    def test_cursor_position_in_bounds(self):
        env = gym.make("neurosim/CursorControl-v0")
        env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            cursor = obs["cursor_position"]
            assert np.all(cursor >= 0.0), f"cursor below lower bound: {cursor}"
            assert np.all(cursor <= 1.0), f"cursor above upper bound: {cursor}"
            if terminated or truncated:
                break
        env.close()

    def test_multiple_steps_episode(self):
        env = gym.make("neurosim/CursorControl-v0")
        env.reset(seed=42)
        truncated = False
        for step_i in range(2000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, (int, float))
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        assert truncated, (
            "Episode should truncate after exhausting all trials"
        )
        env.close()
