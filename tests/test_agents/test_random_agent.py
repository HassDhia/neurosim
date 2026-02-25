"""Tests for RandomAgent baseline."""
import gymnasium as gym
import numpy as np
import pytest

from neurosim.agents.random_agent import RandomAgent


ENV_IDS = [
    "neurosim/DecoderAdapt-v0",
    "neurosim/CursorControl-v0",
    "neurosim/SpellerNav-v0",
]


class TestRandomAgent:
    def test_predict_returns_valid_action(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        agent = RandomAgent(env, seed=42)
        obs, _ = env.reset(seed=42)
        action, info = agent.predict(obs)
        assert isinstance(info, dict)
        assert env.action_space.contains(action)
        env.close()

    def test_evaluate_returns_stats(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        agent = RandomAgent(env, seed=42)
        stats = agent.evaluate(n_episodes=3)
        assert "mean_reward" in stats
        assert "std_reward" in stats
        assert "min_reward" in stats
        assert "max_reward" in stats
        assert isinstance(stats["mean_reward"], float)
        assert stats["min_reward"] <= stats["mean_reward"] <= stats["max_reward"]
        env.close()

    def test_deterministic_with_seed(self):
        """Two agents with the same seed produce identical RNG sequences."""
        env1 = gym.make("neurosim/CursorControl-v0")
        env2 = gym.make("neurosim/CursorControl-v0")
        agent1 = RandomAgent(env1, seed=99)
        agent2 = RandomAgent(env2, seed=99)
        # The agent stores a seeded numpy rng; verify sequences match
        seq1 = agent1.rng.random(10)
        seq2 = agent2.rng.random(10)
        np.testing.assert_array_equal(seq1, seq2)
        env1.close()
        env2.close()

    @pytest.mark.parametrize("env_id", ENV_IDS)
    def test_works_on_all_envs(self, env_id):
        env = gym.make(env_id)
        agent = RandomAgent(env, seed=42)
        obs, _ = env.reset(seed=42)
        action, info = agent.predict(obs)
        assert env.action_space.contains(action)
        # Take one step to confirm no crash
        env.step(action)
        env.close()
