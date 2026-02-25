"""SpellerNav-v0: P300/SSVEP speller stimulus selection environment.

The agent controls which stimulus groups to flash in a BCI speller and
decides when to commit to a symbol selection.  This models the
active-query BCI paradigm where the system chooses which stimuli to
present rather than cycling through a fixed row/column pattern.

The key trade-off: more flashes improve accuracy but reduce
information transfer rate (ITR).  The agent must balance evidence
accumulation against communication speed.

Observation:
    posterior          (36,)      -- Bayesian posterior over symbols
    response_history   (10, 8)   -- last N ERP responses (n_history x n_channels)
    n_flashes          (1,)      -- flashes used so far (normalised)
    target_queue       (5,)      -- upcoming target symbols (one-hot indices)
    snr_estimate       (1,)      -- estimated signal-to-noise ratio

Action (Dict):
    stimulus_group  Discrete(6)  -- which group of symbols to flash
    commit          Discrete(2)  -- 0 = keep querying, 1 = commit to MAP symbol
"""
from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from neurosim.models.pipeline import SignalPipeline


class SpellerNavEnv(gym.Env):
    """P300/SSVEP speller stimulus selection environment.

    The agent selects which stimulus group to flash and decides when
    enough evidence has accumulated to commit to a symbol.  The goal is
    to maximise information transfer rate (bits/min) by spelling a
    target word as fast and accurately as possible.
    """

    metadata: dict[str, Any] = {
        "render_modes": ["human", "ansi"],
        "render_fps": 4,
    }

    # Reward shaping constants
    FLASH_COST: float = 0.02
    CORRECT_COMMIT_BONUS: float = 3.0
    WRONG_COMMIT_PENALTY: float = -2.0
    MAX_FLASHES_PER_SYMBOL: int = 30

    def __init__(
        self,
        n_symbols: int = 36,
        n_groups: int = 6,
        n_channels: int = 8,
        word_length: int = 5,
        n_history: int = 10,
        render_mode: str | None = None,
        p300_amplitude: float = 1.5,
        noise_std: float = 1.0,
        signal_pipeline: SignalPipeline | None = None,
    ) -> None:
        """Initialise the SpellerNav environment.

        Args:
            n_symbols: Size of the symbol alphabet (e.g., 36 = A-Z + 0-9).
            n_groups: Number of stimulus groups (symbols per group = n_symbols / n_groups).
            n_channels: Number of EEG channels.
            word_length: Number of symbols in the target word (queue length).
            n_history: Number of past responses kept in observation.
            render_mode: Gymnasium render mode.
            p300_amplitude: Amplitude of simulated P300 response.
            noise_std: Standard deviation of EEG noise.
            signal_pipeline: Optional signal processing pipeline for pluggable
                drift, noise, and co-adaptation models. When ``None`` (default),
                the environment uses its built-in inline noise generation.
        """
        super().__init__()

        assert n_symbols % n_groups == 0, (
            f"n_symbols ({n_symbols}) must be divisible by n_groups ({n_groups})"
        )
        self._signal_pipeline = signal_pipeline

        self.n_symbols = n_symbols
        self.n_groups = n_groups
        self.n_channels = n_channels
        self.word_length = word_length
        self.n_history = n_history
        self.render_mode = render_mode
        self.p300_amplitude = p300_amplitude
        self.noise_std = noise_std
        self.symbols_per_group = n_symbols // n_groups

        # --- Observation Space ---
        self.observation_space = spaces.Dict(
            {
                "posterior": spaces.Box(
                    low=0.0, high=1.0, shape=(n_symbols,), dtype=np.float32
                ),
                "response_history": spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(n_history, n_channels),
                    dtype=np.float32,
                ),
                "n_flashes": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "target_queue": spaces.Box(
                    low=0.0,
                    high=float(n_symbols - 1),
                    shape=(word_length,),
                    dtype=np.float32,
                ),
                "snr_estimate": spaces.Box(
                    low=0.0, high=10.0, shape=(1,), dtype=np.float32
                ),
            }
        )

        # --- Action Space ---
        self.action_space = spaces.Dict(
            {
                "stimulus_group": spaces.Discrete(n_groups),
                "commit": spaces.Discrete(2),
            }
        )

        # --- Internal State ---
        self._posterior: np.ndarray = np.ones(n_symbols, dtype=np.float32) / n_symbols
        self._response_buffer: np.ndarray = np.zeros(
            (n_history, n_channels), dtype=np.float32
        )
        self._target_queue: np.ndarray = np.zeros(word_length, dtype=np.int32)
        self._current_target: int = 0
        self._queue_idx: int = 0
        self._flash_count: int = 0
        self._total_flashes: int = 0
        self._symbols_spelled: int = 0
        self._correct_symbols: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

        # Group membership: which symbols belong to each group
        self._groups: list[np.ndarray] = [
            np.arange(g * self.symbols_per_group, (g + 1) * self.symbols_per_group)
            for g in range(n_groups)
        ]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment for a new spelling session.

        Args:
            seed: Optional RNG seed.
            options: Additional options (unused).

        Returns:
            observation, info
        """
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # Generate target word (random symbol indices)
        self._target_queue = self._rng.integers(
            0, self.n_symbols, size=self.word_length
        ).astype(np.int32)
        self._queue_idx = 0
        self._current_target = int(self._target_queue[0])

        if self._signal_pipeline is not None:
            self._signal_pipeline.reset()

        # Reset state
        self._posterior = np.ones(self.n_symbols, dtype=np.float32) / self.n_symbols
        self._response_buffer = np.zeros(
            (self.n_history, self.n_channels), dtype=np.float32
        )
        self._flash_count = 0
        self._total_flashes = 0
        self._symbols_spelled = 0
        self._correct_symbols = 0

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one flash/commit decision.

        Args:
            action: Dict with ``stimulus_group`` and ``commit`` keys.

        Returns:
            observation, reward, terminated, truncated, info
        """
        group_idx = int(action["stimulus_group"])
        commit = int(action["commit"])

        reward: float = 0.0
        committed = False
        correct = False

        if commit == 1:
            # --- Commit to MAP symbol ---
            committed = True
            selected = int(np.argmax(self._posterior))
            correct = selected == self._current_target

            if correct:
                reward += self.CORRECT_COMMIT_BONUS
                self._correct_symbols += 1
            else:
                reward += self.WRONG_COMMIT_PENALTY

            # ITR bonus: fewer flashes = higher ITR
            if self._flash_count > 0:
                itr_factor = max(0.0, 1.0 - self._flash_count / self.MAX_FLASHES_PER_SYMBOL)
                reward += itr_factor * 1.0

            self._symbols_spelled += 1
            self._queue_idx += 1

            # Advance to next symbol or end
            if self._queue_idx < self.word_length:
                self._current_target = int(self._target_queue[self._queue_idx])
                self._posterior = (
                    np.ones(self.n_symbols, dtype=np.float32) / self.n_symbols
                )
                self._flash_count = 0
                self._response_buffer = np.zeros(
                    (self.n_history, self.n_channels), dtype=np.float32
                )
        else:
            # --- Flash stimulus group ---
            reward -= self.FLASH_COST
            self._flash_count += 1
            self._total_flashes += 1

            # Simulate ERP response
            response = self._simulate_erp(group_idx)

            # Update response buffer (FIFO)
            self._response_buffer = np.roll(self._response_buffer, -1, axis=0)
            self._response_buffer[-1] = response

            # Bayesian posterior update
            self._update_posterior(group_idx, response)

            # Force commit if too many flashes
            if self._flash_count >= self.MAX_FLASHES_PER_SYMBOL:
                selected = int(np.argmax(self._posterior))
                correct = selected == self._current_target
                committed = True
                if correct:
                    reward += self.CORRECT_COMMIT_BONUS * 0.5  # reduced bonus
                    self._correct_symbols += 1
                else:
                    reward += self.WRONG_COMMIT_PENALTY

                self._symbols_spelled += 1
                self._queue_idx += 1
                if self._queue_idx < self.word_length:
                    self._current_target = int(self._target_queue[self._queue_idx])
                    self._posterior = (
                        np.ones(self.n_symbols, dtype=np.float32) / self.n_symbols
                    )
                    self._flash_count = 0

        terminated = False
        truncated = self._queue_idx >= self.word_length

        obs = self._build_observation()
        info = self._build_info()
        info["committed"] = committed
        info["correct"] = correct

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def _simulate_erp(self, group_idx: int) -> np.ndarray:
        """Simulate an ERP response for a stimulus group.

        If the target symbol is in the flashed group, a P300-like
        response is generated; otherwise only noise.

        Args:
            group_idx: Index of the stimulus group being flashed.

        Returns:
            ERP response array of shape (n_channels,).
        """
        noise = self._rng.standard_normal(self.n_channels).astype(np.float32) * self.noise_std
        target_in_group = self._current_target in self._groups[group_idx]

        if target_in_group:
            # P300 response: positive deflection across channels with spatial pattern
            spatial_pattern = np.array(
                [0.3, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.3][: self.n_channels],
                dtype=np.float32,
            )
            if len(spatial_pattern) < self.n_channels:
                spatial_pattern = np.pad(
                    spatial_pattern,
                    (0, self.n_channels - len(spatial_pattern)),
                    constant_values=0.3,
                )
            signal = self.p300_amplitude * spatial_pattern
            response = signal + noise
        else:
            response = noise

        # Apply optional signal pipeline (drift, noise, co-adaptation)
        if self._signal_pipeline is not None:
            response = self._signal_pipeline.apply(response, self._total_flashes)

        return response

    def _update_posterior(self, group_idx: int, response: np.ndarray) -> None:
        """Bayesian update of symbol posterior given flash response.

        Uses a simplified likelihood model: symbols in the flashed group
        have higher likelihood if response energy is high (P300-like).

        Args:
            group_idx: Index of the flashed stimulus group.
            response: ERP response array of shape (n_channels,).
        """
        response_energy = float(np.mean(response))

        # Likelihood ratio: symbols in the group are more likely if
        # response is large, less likely otherwise
        likelihood = np.ones(self.n_symbols, dtype=np.float32)
        in_group = self._groups[group_idx]

        if response_energy > 0.5:
            # Evidence FOR symbols in this group
            likelihood[in_group] *= 1.0 + 0.3 * response_energy
        else:
            # Evidence AGAINST symbols in this group
            likelihood[in_group] *= max(0.1, 1.0 + 0.3 * response_energy)

        self._posterior *= likelihood
        total = self._posterior.sum()
        if total > 1e-10:
            self._posterior /= total
        else:
            self._posterior = (
                np.ones(self.n_symbols, dtype=np.float32) / self.n_symbols
            )

    # ------------------------------------------------------------------
    # Observation / info builders
    # ------------------------------------------------------------------

    def _build_observation(self) -> dict[str, np.ndarray]:
        """Construct the current observation dict."""
        # SNR estimate from response buffer
        signal_power = float(np.mean(np.abs(self._response_buffer)))
        snr = signal_power / max(self.noise_std, 1e-6)

        return {
            "posterior": self._posterior.copy(),
            "response_history": self._response_buffer.copy(),
            "n_flashes": np.array(
                [self._flash_count / self.MAX_FLASHES_PER_SYMBOL],
                dtype=np.float32,
            ),
            "target_queue": self._target_queue.astype(np.float32),
            "snr_estimate": np.array([np.clip(snr, 0.0, 10.0)], dtype=np.float32),
        }

    def _build_info(self) -> dict[str, Any]:
        """Construct the info dict."""
        return {
            "current_target": self._current_target,
            "queue_idx": self._queue_idx,
            "flash_count": self._flash_count,
            "total_flashes": self._total_flashes,
            "symbols_spelled": self._symbols_spelled,
            "correct_symbols": self._correct_symbols,
            "accuracy": (
                self._correct_symbols / max(self._symbols_spelled, 1)
            ),
            "map_symbol": int(np.argmax(self._posterior)),
            "map_confidence": float(np.max(self._posterior)),
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
            map_sym = int(np.argmax(self._posterior))
            map_conf = float(np.max(self._posterior))
            acc = self._correct_symbols / max(self._symbols_spelled, 1)

            # Symbol display (A-Z for 0-25, 0-9 for 26-35)
            def sym_char(idx: int) -> str:
                if idx < 26:
                    return chr(65 + idx)
                return str(idx - 26)

            target_word = "".join(sym_char(int(t)) for t in self._target_queue)
            progress = "".join(
                sym_char(int(self._target_queue[i]))
                if i < self._queue_idx
                else "_"
                for i in range(self.word_length)
            )

            lines = [
                f"Target: {target_word}  Progress: {progress}",
                f"  Symbol {self._queue_idx + 1}/{self.word_length}  "
                f"Flashes: {self._flash_count}/{self.MAX_FLASHES_PER_SYMBOL}",
                f"  MAP symbol: {sym_char(map_sym)} ({map_conf:.3f})",
                f"  Target:     {sym_char(self._current_target)}",
                f"  Accuracy:   {acc:.2f}  ({self._correct_symbols}/{self._symbols_spelled})",
                f"  Total flashes: {self._total_flashes}",
            ]
            return "\n".join(lines)
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass
