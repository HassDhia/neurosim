"""Action-space wrappers for SB3 compatibility.

SB3's PPO cannot handle gymnasium Dict action spaces.  These wrappers
flatten Dict actions into Box or MultiDiscrete spaces that SB3 supports,
then reconstruct the Dict on the way back to the env.

Usage::

    import gymnasium as gym
    from neurosim.envs.wrappers import wrap_for_sb3

    env = gym.make("neurosim/DecoderAdapt-v0")
    env = wrap_for_sb3(env)  # now has Box/MultiDiscrete action space
"""
from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlattenDictActionWrapper(gym.ActionWrapper):
    """Flatten a Dict action space for SB3 compatibility.

    Handles two patterns found in NeuroSim environments:

    1. **Mixed discrete + continuous** (DecoderAdapt-v0):
       Converts to a single ``Box`` where discrete actions are encoded as
       continuous values and rounded back to ints on step.

    2. **Pure discrete** (SpellerNav-v0):
       Converts to ``MultiDiscrete``.

    The wrapper preserves the original Dict key ordering so it can
    reconstruct the Dict action for the underlying environment.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        assert isinstance(env.action_space, spaces.Dict), (
            f"FlattenDictActionWrapper requires Dict action space, "
            f"got {type(env.action_space)}"
        )

        self._keys: list[str] = list(env.action_space.spaces.keys())
        self._subspaces: dict[str, spaces.Space] = dict(env.action_space.spaces)
        self._has_continuous = any(
            isinstance(s, spaces.Box) for s in self._subspaces.values()
        )

        if self._has_continuous:
            # Mixed mode: flatten everything into a single Box
            self._mode = "box"
            self._slices: dict[str, tuple[int, int]] = {}
            dim = 0
            lows: list[float] = []
            highs: list[float] = []
            for key in self._keys:
                sub = self._subspaces[key]
                if isinstance(sub, spaces.Discrete):
                    size = 1
                    lows.append(0.0)
                    highs.append(float(sub.n - 1))
                elif isinstance(sub, spaces.Box):
                    size = int(np.prod(sub.shape))
                    lows.extend(sub.low.flatten().tolist())
                    highs.extend(sub.high.flatten().tolist())
                else:
                    raise TypeError(f"Unsupported sub-space type: {type(sub)}")
                self._slices[key] = (dim, dim + size)
                dim += size

            self.action_space = spaces.Box(
                low=np.array(lows, dtype=np.float32),
                high=np.array(highs, dtype=np.float32),
                shape=(dim,),
                dtype=np.float32,
            )
        else:
            # Pure discrete: use MultiDiscrete
            self._mode = "multidiscrete"
            nvec = []
            for key in self._keys:
                sub = self._subspaces[key]
                assert isinstance(sub, spaces.Discrete), (
                    f"Expected Discrete, got {type(sub)} for key '{key}'"
                )
                nvec.append(sub.n)
            self.action_space = spaces.MultiDiscrete(np.array(nvec, dtype=np.int64))

    def action(self, action: np.ndarray) -> dict[str, Any]:
        """Convert flat action back to Dict for the underlying env."""
        result: dict[str, Any] = {}
        if self._mode == "box":
            for key in self._keys:
                lo, hi = self._slices[key]
                sub = self._subspaces[key]
                raw = action[lo:hi]
                if isinstance(sub, spaces.Discrete):
                    result[key] = int(np.clip(np.round(raw[0]), 0, sub.n - 1))
                elif isinstance(sub, spaces.Box):
                    result[key] = np.clip(
                        raw, sub.low.flatten(), sub.high.flatten()
                    ).reshape(sub.shape).astype(sub.dtype)
                else:
                    result[key] = raw
        else:
            # MultiDiscrete
            for i, key in enumerate(self._keys):
                result[key] = int(action[i])
        return result


def wrap_for_sb3(env: gym.Env) -> gym.Env:
    """Wrap a NeuroSim environment for SB3 compatibility.

    - If the action space is a ``Dict``, applies :class:`FlattenDictActionWrapper`.
    - If the action space is already SB3-compatible (``Box``, ``Discrete``,
      ``MultiDiscrete``), returns the environment unchanged.

    Args:
        env: A Gymnasium environment.

    Returns:
        The (possibly wrapped) environment ready for SB3.
    """
    if isinstance(env.action_space, spaces.Dict):
        return FlattenDictActionWrapper(env)
    return env
