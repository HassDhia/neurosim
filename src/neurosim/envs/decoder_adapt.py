"""DecoderAdapt-v0: Motor imagery BCI decoder adaptation environment.

The agent classifies motor imagery from neural features while deciding
when to trigger costly recalibration and how aggressively to adapt the
decoder.  This models the real-world trade-off in closed-loop BCIs where
decoder drift degrades accuracy over time, but recalibration interrupts
the user.

Observation:
    neural_features  (24,)   -- band-power / CSP features from EEG
    decoder_confidence (4,)  -- softmax output of current decoder
    decoder_age      (1,)    -- steps since last recalibration (normalised)
    session_progress (1,)    -- fraction of episode elapsed
    recent_accuracy  (1,)    -- rolling accuracy over last 20 trials
    drift_indicator  (24,)   -- abs difference from calibration distribution

Action (Dict):
    classification   Discrete(4)  -- predicted motor-imagery class
    recalibrate      Discrete(2)  -- 0 = keep decoder, 1 = recalibrate now
    adaptation_rate  Box(1)       -- continuous [0, 1] online learning rate
"""
from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from neurosim.models.pipeline import SignalPipeline


class DecoderAdaptEnv(gym.Env):
    """Motor-imagery BCI decoder adaptation environment.

    The agent receives neural features and must classify the imagined
    movement while managing decoder drift through recalibration and
    online adaptation rate control.
    """

    metadata: dict[str, Any] = {
        "render_modes": ["human", "ansi"],
        "render_fps": 4,
    }

    def __init__(
        self,
        n_classes: int = 4,
        n_features: int = 24,
        episode_length: int = 300,
        dataset: str = "BNCI2014_001",
        render_mode: str | None = None,
        drift_rate: float = 0.005,
        recalibration_cost: float = 0.3,
        streak_bonus_threshold: int = 5,
        signal_pipeline: SignalPipeline | None = None,
    ) -> None:
        """Initialise the DecoderAdapt environment.

        Args:
            n_classes: Number of motor-imagery classes.
            n_features: Dimensionality of the neural feature vector.
            episode_length: Maximum steps per episode.
            dataset: MOABB dataset identifier for feature generation.
            render_mode: Gymnasium render mode (``"human"`` or ``"ansi"``).
            drift_rate: Per-step feature distribution drift magnitude.
            recalibration_cost: Reward penalty for triggering recalibration.
            streak_bonus_threshold: Consecutive correct answers before streak bonus.
            signal_pipeline: Optional signal processing pipeline for pluggable
                drift, noise, and co-adaptation models. When ``None`` (default),
                the environment uses its built-in inline noise generation.
        """
        super().__init__()

        self.n_classes = n_classes
        self.n_features = n_features
        self.episode_length = episode_length
        self.dataset = dataset
        self.render_mode = render_mode
        self.drift_rate = drift_rate
        self.recalibration_cost = recalibration_cost
        self.streak_bonus_threshold = streak_bonus_threshold
        self._signal_pipeline = signal_pipeline

        # --- Observation Space ---
        self.observation_space = spaces.Dict(
            {
                "neural_features": spaces.Box(
                    low=-5.0, high=5.0, shape=(n_features,), dtype=np.float32
                ),
                "decoder_confidence": spaces.Box(
                    low=0.0, high=1.0, shape=(n_classes,), dtype=np.float32
                ),
                "decoder_age": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "session_progress": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "recent_accuracy": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "drift_indicator": spaces.Box(
                    low=0.0, high=10.0, shape=(n_features,), dtype=np.float32
                ),
            }
        )

        # --- Action Space ---
        self.action_space = spaces.Dict(
            {
                "classification": spaces.Discrete(n_classes),
                "recalibrate": spaces.Discrete(2),
                "adaptation_rate": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )

        # --- Internal State ---
        self._step_count: int = 0
        self._true_label: int = 0
        self._correct_streak: int = 0
        self._recent_results: list[bool] = []
        self._calibration_mean: np.ndarray = np.zeros(n_features, dtype=np.float32)
        self._current_mean: np.ndarray = np.zeros(n_features, dtype=np.float32)
        self._decoder_age_steps: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment to a fresh session.

        Args:
            seed: Optional RNG seed for reproducibility.
            options: Additional reset options (unused).

        Returns:
            observation: Initial observation dict.
            info: Auxiliary information dict.
        """
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._correct_streak = 0
        self._recent_results = []
        self._decoder_age_steps = 0

        if self._signal_pipeline is not None:
            self._signal_pipeline.reset()

        # Initialise calibration distribution
        self._calibration_mean = self._rng.standard_normal(self.n_features).astype(
            np.float32
        )
        self._current_mean = self._calibration_mean.copy()

        # Sample first trial
        self._true_label = int(self._rng.integers(0, self.n_classes))

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one trial step.

        Args:
            action: Dict with keys ``classification``, ``recalibrate``,
                    ``adaptation_rate``.

        Returns:
            observation, reward, terminated, truncated, info
        """
        classification = int(action["classification"])
        recalibrate = int(action["recalibrate"])
        adaptation_rate = float(np.clip(action["adaptation_rate"], 0.0, 1.0).item())

        self._step_count += 1
        self._decoder_age_steps += 1

        # --- Recalibration ---
        reward: float = 0.0
        if recalibrate == 1:
            self._calibration_mean = self._current_mean.copy()
            self._decoder_age_steps = 0
            reward -= self.recalibration_cost

        # --- Classification Reward ---
        correct = classification == self._true_label
        self._recent_results.append(correct)
        if len(self._recent_results) > 20:
            self._recent_results.pop(0)

        if correct:
            reward += 1.0
            self._correct_streak += 1
            if self._correct_streak >= self.streak_bonus_threshold:
                reward += 0.2  # streak bonus
        else:
            self._correct_streak = 0

        # --- Confidence Penalty ---
        # Penalise low-confidence correct and high-confidence incorrect
        # (placeholder -- real version would use decoder output)
        confidence = 1.0 / self.n_classes  # uniform prior placeholder
        if not correct:
            reward -= 0.1 * confidence

        # --- Drift simulation ---
        drift = self._rng.standard_normal(self.n_features).astype(np.float32)
        self._current_mean += self.drift_rate * drift

        # --- Online adaptation (shifts calibration toward current) ---
        self._calibration_mean += adaptation_rate * 0.01 * (
            self._current_mean - self._calibration_mean
        )

        # --- Next trial ---
        self._true_label = int(self._rng.integers(0, self.n_classes))

        terminated = False
        truncated = self._step_count >= self.episode_length

        obs = self._build_observation()
        info = self._build_info()
        info["correct"] = correct
        info["recalibrated"] = recalibrate == 1

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> dict[str, np.ndarray]:
        """Construct the current observation dict."""
        # Neural features: class-conditioned mean + noise + drift
        class_offset = np.zeros(self.n_features, dtype=np.float32)
        class_offset[self._true_label :: self.n_classes] = 1.0
        neural = (
            self._current_mean
            + class_offset
            + self._rng.standard_normal(self.n_features).astype(np.float32) * 0.5
        )

        # Apply optional signal pipeline (drift, noise, co-adaptation)
        if self._signal_pipeline is not None:
            neural = self._signal_pipeline.apply(neural, self._step_count)

        # Simulated decoder confidence (placeholder softmax)
        logits = self._rng.standard_normal(self.n_classes).astype(np.float32)
        logits[self._true_label] += 1.5  # signal
        exp_logits = np.exp(logits - logits.max())
        confidence = exp_logits / exp_logits.sum()

        # Drift indicator
        drift_indicator = np.abs(self._current_mean - self._calibration_mean)

        # Recent accuracy
        if self._recent_results:
            recent_acc = float(np.mean(self._recent_results))
        else:
            recent_acc = 0.0

        return {
            "neural_features": np.clip(neural, -5.0, 5.0).astype(np.float32),
            "decoder_confidence": confidence.astype(np.float32),
            "decoder_age": np.array(
                [min(self._decoder_age_steps / self.episode_length, 1.0)],
                dtype=np.float32,
            ),
            "session_progress": np.array(
                [self._step_count / self.episode_length], dtype=np.float32
            ),
            "recent_accuracy": np.array([recent_acc], dtype=np.float32),
            "drift_indicator": np.clip(drift_indicator, 0.0, 10.0).astype(
                np.float32
            ),
        }

    def _build_info(self) -> dict[str, Any]:
        """Construct the info dict."""
        return {
            "step": self._step_count,
            "true_label": self._true_label,
            "decoder_age_steps": self._decoder_age_steps,
            "streak": self._correct_streak,
            "dataset": self.dataset,
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
            acc = (
                f"{np.mean(self._recent_results):.2f}"
                if self._recent_results
                else "N/A"
            )
            lines = [
                f"Step {self._step_count}/{self.episode_length}",
                f"  True label:    {self._true_label}",
                f"  Decoder age:   {self._decoder_age_steps}",
                f"  Streak:        {self._correct_streak}",
                f"  Recent acc:    {acc}",
                f"  Mean drift:    {np.mean(np.abs(self._current_mean - self._calibration_mean)):.4f}",
            ]
            return "\n".join(lines)
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass
