"""Comprehensive tests for noise injection models."""

import numpy as np
import pytest

from neurosim.models.noise import NoiseInjector


SEED = 42
SHAPE_2D = (4, 100)
SFREQ = 250.0


@pytest.fixture
def injector():
    return NoiseInjector(sfreq=SFREQ, seed=SEED)


@pytest.fixture
def clean_signal():
    rng = np.random.default_rng(0)
    return rng.standard_normal(SHAPE_2D) * 1e-5


class TestPinkNoise:
    """Tests for 1/f noise injection."""

    def test_pink_noise_changes_signal(self, injector, clean_signal):
        """Output should differ from input after pink noise addition."""
        noisy = injector.add_pink_noise(clean_signal, amplitude=2e-6)
        assert not np.allclose(noisy, clean_signal)

    def test_pink_noise_spectral_shape(self, injector):
        """Pink noise should have more low-frequency power than high-frequency."""
        # Use a longer signal for cleaner spectrum
        signal = np.zeros((1, 1024))
        noisy = injector.add_pink_noise(signal, amplitude=1.0)
        noise = noisy[0]  # extract the noise (signal was zeros)

        spectrum = np.abs(np.fft.rfft(noise))
        freqs = np.fft.rfftfreq(1024, d=1.0 / SFREQ)

        # Compare low-freq band (1-10 Hz) vs high-freq band (50-100 Hz)
        low_mask = (freqs >= 1) & (freqs <= 10)
        high_mask = (freqs >= 50) & (freqs <= 100)

        low_power = np.mean(spectrum[low_mask] ** 2)
        high_power = np.mean(spectrum[high_mask] ** 2)

        assert low_power > high_power, (
            f"Pink noise should have more low-freq power ({low_power:.4f}) "
            f"than high-freq ({high_power:.4f})"
        )


class TestEMGArtifact:
    """Tests for EMG burst artifacts."""

    def test_emg_artifact_only_at_bursts(self, injector, clean_signal):
        """With p=0, signal unchanged; with p=1, signal modified everywhere."""
        # p=0: no bursts
        unchanged = injector.add_emg_artifact(clean_signal, p=0.0)
        np.testing.assert_array_equal(unchanged, clean_signal)

        # reset RNG for clean state
        injector.reset()

        # p=1: every sample gets a burst
        changed = injector.add_emg_artifact(clean_signal, p=1.0)
        assert not np.allclose(changed, clean_signal)


class TestEyeBlink:
    """Tests for eye blink artifacts."""

    def test_eye_blink_frontal_stronger(self):
        """Front channels should be more affected than rear channels."""
        # Use many channels to see the gradient clearly
        n_ch = 8
        signal = np.zeros((n_ch, 500))
        injector = NoiseInjector(sfreq=SFREQ, seed=SEED)
        # High blink probability to guarantee blinks occur
        blinked = injector.add_eye_blink(signal, p=1.0)

        # Measure change per channel
        channel_energy = np.mean(np.abs(blinked), axis=1)

        # Front channels (index 0) should have more energy than rear (index -1)
        # The attenuation formula: max(0, 1 - ch/n_ch * 2) means ch >= n_ch/2 get 0
        front_energy = channel_energy[0]
        rear_energy = channel_energy[-1]
        assert front_energy > rear_energy, (
            f"Front channel energy ({front_energy:.6f}) should exceed "
            f"rear channel energy ({rear_energy:.6f})"
        )


class TestLineNoise:
    """Tests for power line interference."""

    def test_line_noise_frequency(self, injector):
        """Dominant frequency should be at 60 Hz."""
        # Use clean zeros so line noise is the only contribution
        n_samples = 2048
        signal = np.zeros((1, n_samples))
        noisy = injector.add_line_noise(signal, freq=60.0, amplitude=1.0)
        noise = noisy[0]

        spectrum = np.abs(np.fft.rfft(noise))
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / SFREQ)

        peak_freq = freqs[np.argmax(spectrum[1:])] if freqs[0] == 0 else freqs[np.argmax(spectrum)]
        # The peak should be within one frequency bin of 60 Hz
        freq_resolution = SFREQ / n_samples
        assert abs(peak_freq - 60.0) <= freq_resolution * 2, (
            f"Peak frequency {peak_freq:.2f} Hz, expected ~60 Hz"
        )


class TestInjectPresets:
    """Tests for combined inject() with level presets."""

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_inject_preset_low_medium_high(self, injector, clean_signal, level):
        """All 3 presets run without error."""
        injector.reset()
        result = injector.inject(clean_signal, noise_level=level)
        assert result.shape == clean_signal.shape
        assert not np.allclose(result, clean_signal)

    def test_inject_progressive_noise(self, clean_signal):
        """Higher preset levels add progressively more noise."""
        diffs = {}
        for level in ["low", "medium", "high"]:
            inj = NoiseInjector(sfreq=SFREQ, seed=SEED)
            result = inj.inject(clean_signal, noise_level=level)
            diffs[level] = np.mean(np.abs(result - clean_signal))

        assert diffs["low"] < diffs["medium"] < diffs["high"], (
            f"Expected low < medium < high noise, got: "
            f"low={diffs['low']:.8f}, medium={diffs['medium']:.8f}, high={diffs['high']:.8f}"
        )

    def test_inject_invalid_level_raises(self, injector, clean_signal):
        """Invalid level raises ValueError."""
        with pytest.raises(ValueError, match="Unknown noise_level"):
            injector.inject(clean_signal, noise_level="extreme")


class TestNoiseInjectorReset:
    """Tests for RNG reset reproducibility."""

    def test_noise_injector_reset_reproducible(self, clean_signal):
        """Reset + apply produces the same result."""
        inj = NoiseInjector(sfreq=SFREQ, seed=SEED)
        first = inj.inject(clean_signal, noise_level="medium")
        inj.reset()
        second = inj.inject(clean_signal, noise_level="medium")
        np.testing.assert_array_equal(first, second)
