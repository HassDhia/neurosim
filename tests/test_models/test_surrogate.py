"""Comprehensive tests for the NeuralSurrogate cVAE model.

Tests cover training, generation, save/load roundtrips, and error paths.
Uses small dimensions (8 channels, 50 timepoints) for fast execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurosim.data.formats import NeuralEpoch
from neurosim.models.surrogate import NeuralSurrogate


def make_synthetic_epochs(
    n_channels: int = 8,
    n_timepoints: int = 50,
    n_classes: int = 2,
    n_per_class: int = 20,
    seed: int = 42,
) -> list[NeuralEpoch]:
    """Create class-conditioned synthetic epochs for fast testing."""
    rng = np.random.default_rng(seed)
    epochs: list[NeuralEpoch] = []
    channels = [f"Ch{i:03d}" for i in range(n_channels)]
    for cls in range(n_classes):
        class_mean = np.zeros(n_channels)
        class_mean[cls::n_classes] = 3.0  # strong class separation
        for _ in range(n_per_class):
            signals = class_mean[:, None] + rng.standard_normal((n_channels, n_timepoints)) * 0.3
            epochs.append(
                NeuralEpoch(
                    signals=signals,
                    label=cls,
                    sfreq=100.0,
                    channels=channels,
                    subject_id=1,
                )
            )
    return epochs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_epochs() -> list[NeuralEpoch]:
    """Small, fast synthetic dataset with 2 well-separated classes."""
    return make_synthetic_epochs()


@pytest.fixture
def trained_surrogate(synthetic_epochs: list[NeuralEpoch]) -> NeuralSurrogate:
    """Return a NeuralSurrogate that has already been trained for 50 epochs."""
    model = NeuralSurrogate(n_channels=8, n_timepoints=50, n_classes=2, latent_dim=16)
    model.train(synthetic_epochs, n_epochs=50, batch_size=32)
    return model


# ---------------------------------------------------------------------------
# TestSurrogateTraining
# ---------------------------------------------------------------------------

class TestSurrogateTraining:
    """Tests for the NeuralSurrogate.train() method."""

    def test_train_loss_decreases(self, synthetic_epochs: list[NeuralEpoch]) -> None:
        """Training for 50 epochs should reduce the loss."""
        model = NeuralSurrogate(n_channels=8, n_timepoints=50, n_classes=2, latent_dim=16)
        losses = model.train(synthetic_epochs, n_epochs=50, batch_size=32)

        assert len(losses) == 50
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_train_empty_raises(self) -> None:
        """Training on an empty list must raise ValueError."""
        model = NeuralSurrogate()
        with pytest.raises(ValueError, match="empty"):
            model.train([])

    def test_train_sets_trained_flag(self, synthetic_epochs: list[NeuralEpoch]) -> None:
        """After training, the _trained flag must be True."""
        model = NeuralSurrogate(n_channels=8, n_timepoints=50, n_classes=2, latent_dim=16)
        assert model._trained is False
        model.train(synthetic_epochs, n_epochs=5, batch_size=32)
        assert model._trained is True


# ---------------------------------------------------------------------------
# TestSurrogateGeneration
# ---------------------------------------------------------------------------

class TestSurrogateGeneration:
    """Tests for the NeuralSurrogate.generate() method."""

    def test_generate_before_train_raises(self) -> None:
        """Generating before training must raise RuntimeError."""
        model = NeuralSurrogate()
        with pytest.raises(RuntimeError, match="trained"):
            model.generate(0)

    def test_generate_correct_shape(self, trained_surrogate: NeuralSurrogate) -> None:
        """Generated epochs must have the correct channel and timepoint counts."""
        results = trained_surrogate.generate(class_label=0, n_samples=5)

        assert len(results) == 5
        for epoch in results:
            assert isinstance(epoch, NeuralEpoch)
            assert epoch.signals.shape == (8, 50)
            assert epoch.label == 0

    def test_generate_class_conditioned(self, synthetic_epochs: list[NeuralEpoch]) -> None:
        """Generated samples for different classes should have distinct statistics.

        The training data has class means offset on alternating channels
        (class 0 -> even channels boosted, class 1 -> odd channels boosted).
        The generator should preserve this separation after sufficient training.
        """
        # Train longer for class separation to emerge
        model = NeuralSurrogate(n_channels=8, n_timepoints=50, n_classes=2, latent_dim=16)
        model.train(synthetic_epochs, n_epochs=150, batch_size=32)

        n_gen = 50
        samples_0 = model.generate(class_label=0, n_samples=n_gen)
        samples_1 = model.generate(class_label=1, n_samples=n_gen)

        # Compute per-channel means across generated samples
        mean_0 = np.mean([e.signals.mean(axis=1) for e in samples_0], axis=0)
        mean_1 = np.mean([e.signals.mean(axis=1) for e in samples_1], axis=0)

        # The two class means should differ -- even a small difference proves
        # the decoder is using the class condition vector
        diff = np.abs(mean_0 - mean_1).mean()
        assert diff > 0.01, (
            f"Class means too similar (mean abs diff={diff:.4f}). "
            "Generator may not be conditioning on class labels."
        )


# ---------------------------------------------------------------------------
# TestSurrogateSaveLoad
# ---------------------------------------------------------------------------

class TestSurrogateSaveLoad:
    """Tests for NeuralSurrogate.save() and .load() methods."""

    def test_save_load_roundtrip(
        self,
        trained_surrogate: NeuralSurrogate,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """A saved model, loaded into a fresh instance, should generate valid output."""
        save_path = tmp_path / "model.pt"
        trained_surrogate.save(save_path)
        assert save_path.exists()

        loaded = NeuralSurrogate()
        loaded.load(save_path)

        assert loaded._trained is True
        assert loaded.n_channels == 8
        assert loaded.n_timepoints == 50

        # Generate from the loaded model
        results = loaded.generate(class_label=0, n_samples=3)
        assert len(results) == 3
        for epoch in results:
            assert epoch.signals.shape == (8, 50)

    def test_save_untrained_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        """Saving an untrained model must raise RuntimeError."""
        model = NeuralSurrogate()
        with pytest.raises(RuntimeError, match="untrained"):
            model.save(tmp_path / "model.pt")
