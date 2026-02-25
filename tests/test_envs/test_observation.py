"""Tests for ObservationFlattener utility."""
import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from neurosim.envs.observation import ObservationFlattener


class TestObservationFlattener:
    def test_from_env_obs_space(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        flattener = ObservationFlattener(env.observation_space)
        assert flattener.flat_dim > 0
        env.close()

    def test_flatten_produces_1d_array(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        flattener = ObservationFlattener(env.observation_space)
        obs, _ = env.reset(seed=42)
        flat = flattener.flatten(obs)
        assert flat.ndim == 1
        assert flat.dtype == np.float32
        assert flat.shape[0] == flattener.flat_dim
        env.close()

    def test_unflatten_reconstructs_dict_keys(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        flattener = ObservationFlattener(env.observation_space)
        obs, _ = env.reset(seed=42)
        flat = flattener.flatten(obs)
        reconstructed = flattener.unflatten(flat)
        assert isinstance(reconstructed, dict)
        assert set(reconstructed.keys()) == set(obs.keys())
        env.close()

    def test_roundtrip_preserves_values(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        flattener = ObservationFlattener(env.observation_space)
        obs, _ = env.reset(seed=42)
        flat = flattener.flatten(obs)
        reconstructed = flattener.unflatten(flat)
        for key in obs:
            np.testing.assert_allclose(
                reconstructed[key].astype(np.float32),
                obs[key].astype(np.float32),
                atol=1e-6,
            )
        env.close()

    def test_flat_dim_matches_total_obs_size(self):
        env = gym.make("neurosim/DecoderAdapt-v0")
        flattener = ObservationFlattener(env.observation_space)
        expected = sum(
            int(np.prod(s.shape))
            for s in env.observation_space.spaces.values()
        )
        assert flattener.flat_dim == expected
        env.close()

    def test_non_dict_raises_type_error(self):
        box_space = spaces.Box(low=0.0, high=1.0, shape=(10,))
        with pytest.raises(TypeError, match="Expected spaces.Dict"):
            ObservationFlattener(box_space)
