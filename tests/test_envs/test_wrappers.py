"""Tests for action-space wrappers (SB3 compatibility)."""
import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from neurosim.envs.wrappers import FlattenDictActionWrapper, wrap_for_sb3


class TestFlattenDictActionWrapperBoxMode:
    """DecoderAdapt-v0 has Discrete + Box actions -> Box wrapper mode."""

    def test_produces_box_action_space(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        wrapped = FlattenDictActionWrapper(env)
        assert isinstance(wrapped.action_space, spaces.Box)
        wrapped.close()

    def test_step_succeeds_with_sampled_action(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        wrapped = FlattenDictActionWrapper(env)
        obs, _ = wrapped.reset(seed=42)
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)
        assert isinstance(reward, (int, float))
        wrapped.close()

    def test_obs_space_unchanged(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        original_obs_space = env.observation_space
        wrapped = FlattenDictActionWrapper(env)
        assert wrapped.observation_space == original_obs_space
        wrapped.close()

    def test_action_roundtrip(self):
        """A flat action fed through step should not raise."""
        env = gym.make("neurosim/DecoderAdapt-v0")
        wrapped = FlattenDictActionWrapper(env)
        wrapped.reset(seed=42)
        flat_action = wrapped.action_space.sample()
        # The wrapper internally converts flat -> dict -> env.step
        obs, reward, terminated, truncated, info = wrapped.step(flat_action)
        assert isinstance(obs, dict)
        wrapped.close()


class TestFlattenDictActionWrapperMultiDiscreteMode:
    """SpellerNav-v0 has Discrete + Discrete actions -> MultiDiscrete mode."""

    def test_produces_multidiscrete_action_space(self):
        env = gym.make("neurosim/SpellerNav-v0")
        wrapped = FlattenDictActionWrapper(env)
        assert isinstance(wrapped.action_space, spaces.MultiDiscrete)
        wrapped.close()

    def test_action_roundtrip(self):
        env = gym.make("neurosim/SpellerNav-v0")
        wrapped = FlattenDictActionWrapper(env)
        wrapped.reset(seed=42)
        flat_action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(flat_action)
        assert isinstance(obs, dict)
        wrapped.close()


class TestWrapForSB3:
    def test_passthrough_for_box_action_space(self):
        """CursorControl-v0 already has Box actions -- should pass through."""
        env = gym.make("neurosim/CursorControl-v0")
        assert isinstance(env.action_space, spaces.Box)
        wrapped = wrap_for_sb3(env)
        assert wrapped is env  # same object, no wrapping
        env.close()

    def test_wraps_dict_action_space(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        assert isinstance(env.action_space, spaces.Dict)
        wrapped = wrap_for_sb3(env)
        assert not isinstance(wrapped.action_space, spaces.Dict)
        wrapped.close()
