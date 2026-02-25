"""Observation space flattening utilities for NeuroSim environments."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ObservationFlattener:
    """Flattens Dict observation spaces to a single Box for use with standard RL agents.

    Many RL algorithms (PPO, SAC, DQN) expect a flat vector observation.
    This utility converts Dict spaces used by NeuroSim environments into
    a single contiguous Box space, with deterministic key ordering.

    Usage:
        flattener = ObservationFlattener(env.observation_space)
        flat_obs = flattener.flatten(obs_dict)
        reconstructed = flattener.unflatten(flat_obs)
    """

    def __init__(self, observation_space: spaces.Dict) -> None:
        """Initialize the flattener from a Dict observation space.

        Args:
            observation_space: A gymnasium Dict space to flatten.
        """
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(
                f"Expected spaces.Dict, got {type(observation_space).__name__}"
            )
        self.original_space = observation_space
        self._keys: list[str] = sorted(observation_space.spaces.keys())
        self._shapes: dict[str, tuple[int, ...]] = {
            k: observation_space[k].shape for k in self._keys
        }
        self._sizes: dict[str, int] = {
            k: int(np.prod(shape)) for k, shape in self._shapes.items()
        }
        flat_dim = sum(self._sizes.values())
        self.flat_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    @property
    def flat_dim(self) -> int:
        """Total dimensionality of the flattened observation."""
        return self.flat_space.shape[0]

    def flatten(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten a Dict observation into a 1-D float32 array.

        Args:
            obs: Dictionary mapping keys to numpy arrays matching the
                 original observation space.

        Returns:
            A 1-D float32 numpy array of shape (flat_dim,).
        """
        parts = [obs[k].flatten().astype(np.float32) for k in self._keys]
        return np.concatenate(parts)

    def unflatten(self, flat_obs: np.ndarray) -> dict[str, np.ndarray]:
        """Reconstruct a Dict observation from a flat array.

        Args:
            flat_obs: A 1-D array of shape (flat_dim,).

        Returns:
            Dictionary mapping keys to numpy arrays with original shapes.
        """
        result: dict[str, np.ndarray] = {}
        idx = 0
        for k in self._keys:
            size = self._sizes[k]
            result[k] = flat_obs[idx : idx + size].reshape(self._shapes[k])
            idx += size
        return result
