"""Tests for DecoderAdapt-v0 environment."""
import gymnasium as gym
import numpy as np
import pytest

import neurosim.envs  # noqa: F401 — triggers gymnasium env registration


class TestDecoderAdaptEnv:
    def test_env_creation(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        assert env is not None
        env.close()

    def test_reset_returns_obs_and_info(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        env.close()

    def test_step_returns_correct_tuple(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
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
        env = gym.make("neurosim/DecoderAdapt-v0")
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        env.close()

    def test_render_ansi_returns_string(self):
        env = gym.make("neurosim/DecoderAdapt-v0", render_mode="ansi")
        env.reset(seed=42)
        action = env.action_space.sample()
        env.step(action)
        output = env.render()
        assert isinstance(output, str)
        assert len(output) > 0
        env.close()

    def test_info_contains_accuracy_key(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "correct" in info, "step info should contain 'correct' accuracy indicator"
        assert isinstance(info["correct"], bool)
        env.close()

    def test_multiple_steps_episode(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        env.reset(seed=42)
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, (int, float))
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        env.close()

    def test_seed_determinism(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        obs_a, _ = env.reset(seed=123)
        obs_b, _ = env.reset(seed=123)
        for key in obs_a:
            np.testing.assert_array_equal(
                obs_a[key], obs_b[key],
                err_msg=f"Observation key '{key}' differs across same-seed resets",
            )
        env.close()
