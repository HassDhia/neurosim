"""CursorControl-v0: Continuous 2D cursor control via neural decoding.

The agent decodes neural population activity into 2D cursor velocity
commands to acquire screen targets.  This models intracortical BCI
cursor control (e.g., BrainGate-style), where a population of motor
cortex neurons encodes intended movement direction and speed.

Observation:
    neural_state     (96, 5) -- spike counts in time bins (n_units x n_bins)
    cursor_position  (2,)    -- current [x, y] in normalised workspace
    target_position  (2,)    -- target [x, y]
    cursor_velocity  (2,)    -- current velocity vector
    distance_to_target (1,)  -- Euclidean distance to target
    time_in_trial    (1,)    -- fraction of max_trial_steps elapsed

Action:
    Box(2) in [-1, 1] -- decoded velocity command [vx, vy]
"""
from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from neurosim.models.pipeline import SignalPipeline


class CursorControlEnv(gym.Env):
    """Continuous 2D cursor control BCI environment.

    Each episode contains multiple reach trials.  Within a trial the
    agent commands cursor velocity to reach a target.  A trial ends on
    target acquisition (cursor within radius) or timeout.
    """

    metadata: dict[str, Any] = {
        "render_modes": ["human", "ansi"],
        "render_fps": 20,
    }

    # Workspace is normalised to [0, 1] x [0, 1]
    WORKSPACE_LOW: float = 0.0
    WORKSPACE_HIGH: float = 1.0
    ACQUIRE_RADIUS: float = 0.05
    VELOCITY_SCALE: float = 0.05  # max displacement per step

    def __init__(
        self,
        n_units: int = 96,
        n_bins: int = 5,
        max_trial_steps: int = 30,
        trials_per_episode: int = 50,
        render_mode: str | None = None,
        noise_scale: float = 0.3,
        jitter_penalty_weight: float = 0.1,
        signal_pipeline: SignalPipeline | None = None,
    ) -> None:
        """Initialise the CursorControl environment.

        Args:
            n_units: Number of neural units (electrodes / sorted units).
            n_bins: Number of time bins in each observation window.
            max_trial_steps: Maximum steps allowed per reach trial.
            trials_per_episode: Number of reach trials per episode.
            render_mode: Gymnasium render mode.
            noise_scale: Std-dev of neural noise added to tuning curves.
            jitter_penalty_weight: Weight for velocity-jitter penalty.
            signal_pipeline: Optional signal processing pipeline for pluggable
                drift, noise, and co-adaptation models. When ``None`` (default),
                the environment uses its built-in inline noise generation.
        """
        super().__init__()

        self.n_units = n_units
        self.n_bins = n_bins
        self.max_trial_steps = max_trial_steps
        self.trials_per_episode = trials_per_episode
        self.render_mode = render_mode
        self.noise_scale = noise_scale
        self.jitter_penalty_weight = jitter_penalty_weight
        self._signal_pipeline = signal_pipeline

        # --- Observation Space ---
        self.observation_space = spaces.Dict(
            {
                "neural_state": spaces.Box(
                    low=0.0,
                    high=50.0,
                    shape=(n_units, n_bins),
                    dtype=np.float32,
                ),
                "cursor_position": spaces.Box(
                    low=0.0, high=1.0, shape=(2,), dtype=np.float32
                ),
                "target_position": spaces.Box(
                    low=0.0, high=1.0, shape=(2,), dtype=np.float32
                ),
                "cursor_velocity": spaces.Box(
                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                ),
                "distance_to_target": spaces.Box(
                    low=0.0, high=2.0, shape=(1,), dtype=np.float32
                ),
                "time_in_trial": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )

        # --- Action Space ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # --- Internal State ---
        self._cursor_pos: np.ndarray = np.array([0.5, 0.5], dtype=np.float32)
        self._cursor_vel: np.ndarray = np.zeros(2, dtype=np.float32)
        self._target_pos: np.ndarray = np.array([0.5, 0.5], dtype=np.float32)
        self._prev_vel: np.ndarray = np.zeros(2, dtype=np.float32)
        self._trial_step: int = 0
        self._trial_count: int = 0
        self._total_steps: int = 0
        self._acquisitions: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

        # Cosine-tuning preferred directions for each unit
        self._preferred_dirs: np.ndarray = np.zeros((n_units, 2), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment to first trial of a new episode.

        Args:
            seed: Optional RNG seed.
            options: Additional options (unused).

        Returns:
            observation, info
        """
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # Generate random preferred directions for neural tuning
        angles = self._rng.uniform(0, 2 * np.pi, size=self.n_units)
        self._preferred_dirs = np.stack(
            [np.cos(angles), np.sin(angles)], axis=-1
        ).astype(np.float32)

        self._trial_count = 0
        self._total_steps = 0
        self._acquisitions = 0

        if self._signal_pipeline is not None:
            self._signal_pipeline.reset()

        self._start_new_trial()

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Apply velocity command and advance one step.

        Args:
            action: 2-D velocity command in [-1, 1].

        Returns:
            observation, reward, terminated, truncated, info
        """
        action = np.asarray(action, dtype=np.float32).flatten()[:2]
        action = np.clip(action, -1.0, 1.0)

        self._prev_vel = self._cursor_vel.copy()
        self._cursor_vel = action
        self._trial_step += 1
        self._total_steps += 1

        # --- Move cursor ---
        displacement = action * self.VELOCITY_SCALE
        self._cursor_pos = np.clip(
            self._cursor_pos + displacement,
            self.WORKSPACE_LOW,
            self.WORKSPACE_HIGH,
        ).astype(np.float32)

        # --- Distance & progress ---
        prev_dist = float(np.linalg.norm(self._cursor_pos - displacement - self._target_pos))
        curr_dist = float(np.linalg.norm(self._cursor_pos - self._target_pos))

        # --- Reward ---
        reward: float = 0.0

        # Negative distance (encourage being close)
        reward -= curr_dist * 0.1

        # Progress toward target
        progress = prev_dist - curr_dist
        reward += progress * 2.0

        # Jitter penalty (penalise erratic velocity changes)
        jitter = float(np.linalg.norm(self._cursor_vel - self._prev_vel))
        reward -= self.jitter_penalty_weight * jitter

        # --- Target acquisition ---
        acquired = curr_dist < self.ACQUIRE_RADIUS
        trial_timeout = self._trial_step >= self.max_trial_steps

        if acquired:
            time_fraction = self._trial_step / self.max_trial_steps
            reward += 5.0 * (1.0 - 0.5 * time_fraction)  # faster = more reward
            self._acquisitions += 1

        # --- Trial management ---
        if acquired or trial_timeout:
            self._trial_count += 1
            if self._trial_count < self.trials_per_episode:
                self._start_new_trial()

        terminated = False
        truncated = self._trial_count >= self.trials_per_episode

        obs = self._build_observation()
        info = self._build_info()
        info["acquired"] = acquired
        info["trial_timeout"] = trial_timeout

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_new_trial(self) -> None:
        """Reset cursor and sample a new target for the next trial."""
        self._cursor_pos = np.array([0.5, 0.5], dtype=np.float32)
        self._cursor_vel = np.zeros(2, dtype=np.float32)
        self._prev_vel = np.zeros(2, dtype=np.float32)
        self._trial_step = 0

        # Random target on workspace (avoid centre region)
        while True:
            target = self._rng.uniform(0.1, 0.9, size=2).astype(np.float32)
            if np.linalg.norm(target - self._cursor_pos) > 0.2:
                break
        self._target_pos = target

    def _generate_neural_state(self) -> np.ndarray:
        """Generate cosine-tuned neural activity for the current velocity.

        Returns:
            Array of shape (n_units, n_bins) with simulated spike counts.
        """
        # Direction of intended movement = toward target
        intended = self._target_pos - self._cursor_pos
        norm = np.linalg.norm(intended)
        if norm > 1e-6:
            intended = intended / norm
        else:
            intended = np.zeros(2, dtype=np.float32)

        # Cosine tuning: rate = baseline + gain * cos(angle)
        cos_sim = self._preferred_dirs @ intended  # (n_units,)
        baseline = 10.0
        gain = 8.0
        rates = baseline + gain * cos_sim  # (n_units,)
        rates = np.clip(rates, 0.5, 50.0)

        # Generate spike counts across time bins (Poisson-like)
        neural = np.zeros((self.n_units, self.n_bins), dtype=np.float32)
        for b in range(self.n_bins):
            neural[:, b] = (
                rates
                + self._rng.standard_normal(self.n_units).astype(np.float32)
                * self.noise_scale
                * rates
            )

        # Apply optional signal pipeline (drift, noise, co-adaptation)
        if self._signal_pipeline is not None:
            neural = self._signal_pipeline.apply(neural, self._total_steps)

        return np.clip(neural, 0.0, 50.0).astype(np.float32)

    def _build_observation(self) -> dict[str, np.ndarray]:
        """Construct the current observation dict."""
        dist = float(np.linalg.norm(self._cursor_pos - self._target_pos))
        return {
            "neural_state": self._generate_neural_state(),
            "cursor_position": self._cursor_pos.copy(),
            "target_position": self._target_pos.copy(),
            "cursor_velocity": self._cursor_vel.copy(),
            "distance_to_target": np.array([dist], dtype=np.float32),
            "time_in_trial": np.array(
                [self._trial_step / self.max_trial_steps], dtype=np.float32
            ),
        }

    def _build_info(self) -> dict[str, Any]:
        """Construct the info dict."""
        return {
            "trial": self._trial_count,
            "trial_step": self._trial_step,
            "total_steps": self._total_steps,
            "acquisitions": self._acquisitions,
            "acquisition_rate": (
                self._acquisitions / max(self._trial_count, 1)
            ),
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str | None:
        """Render the environment state.

        Returns:
            ANSI string if render_mode is ``"ansi"``, else None.
        """
        if self.render_mode == "ansi":
            dist = float(np.linalg.norm(self._cursor_pos - self._target_pos))
            lines = [
                f"Trial {self._trial_count + 1}/{self.trials_per_episode}  "
                f"Step {self._trial_step}/{self.max_trial_steps}",
                f"  Cursor:   ({self._cursor_pos[0]:.3f}, {self._cursor_pos[1]:.3f})",
                f"  Target:   ({self._target_pos[0]:.3f}, {self._target_pos[1]:.3f})",
                f"  Distance: {dist:.4f}  (acquire < {self.ACQUIRE_RADIUS})",
                f"  Velocity: ({self._cursor_vel[0]:.3f}, {self._cursor_vel[1]:.3f})",
                f"  Acquired: {self._acquisitions}/{self._trial_count}",
            ]
            return "\n".join(lines)
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass
