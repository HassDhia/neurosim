"""Neural surrogate model for synthetic EEG generation.

Implements a conditional Variational Autoencoder (cVAE) that learns to
generate realistic EEG epochs conditioned on class labels and drift state.
This enables infinite training data and controlled non-stationarity experiments.

Architecture:
    Encoder: signal + class_label + drift_state -> latent (mu, logvar)
    Decoder: latent + class_label + drift_state -> reconstructed signal

The model is optional -- torch is only imported when train/generate are called.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from neurosim.data.formats import NeuralEpoch

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)


def _require_torch() -> Any:
    """Import and return the torch module, raising helpful error if missing.

    Returns:
        The torch module.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the NeuralSurrogate model but is not installed.\n"
            "Install it with: pip install torch\n"
            "Or for the full NeuroSim stack: pip install neurosim[models]"
        ) from exc


class NeuralSurrogate:
    """Conditional VAE for generating synthetic BCI epochs.

    Learns the distribution of EEG signals conditioned on class labels
    and optional drift state vectors, enabling generation of realistic
    synthetic data for RL environment training.

    Args:
        n_channels: Number of EEG channels.
        n_timepoints: Number of time samples per epoch.
        latent_dim: Dimensionality of the VAE latent space.
        n_classes: Number of BCI output classes.
        learning_rate: Adam optimizer learning rate.

    Example::

        surrogate = NeuralSurrogate(n_channels=22, n_timepoints=250, n_classes=4)
        surrogate.train(real_epochs, n_epochs=100)
        synthetic = surrogate.generate(class_label=1, n_samples=10)
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_timepoints: int = 250,
        latent_dim: int = 32,
        n_classes: int = 4,
        learning_rate: float = 1e-3,
    ) -> None:
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.learning_rate = learning_rate

        # Model components (initialized lazily when torch is available)
        self._encoder: Any = None
        self._decoder: Any = None
        self._optimizer: Any = None
        self._trained: bool = False
        self._device: str = "cpu"

        # Training metadata
        self._train_losses: list[float] = []
        self._channel_names: list[str] = []
        self._sfreq: float = 250.0

    @property
    def input_dim(self) -> int:
        """Flattened input dimension (n_channels * n_timepoints)."""
        return self.n_channels * self.n_timepoints

    def _build_model(self) -> None:
        """Construct the encoder/decoder networks.

        Called lazily on first train() or load() to avoid importing
        torch at module level.
        """
        torch = _require_torch()
        import torch.nn as nn

        input_dim = self.input_dim
        condition_dim = self.n_classes  # one-hot class label

        # Encoder: input + condition -> latent (mu, logvar)
        self._encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self._enc_mu = nn.Linear(128, self.latent_dim)
        self._enc_logvar = nn.Linear(128, self.latent_dim)

        # Decoder: latent + condition -> reconstructed input
        self._decoder = nn.Sequential(
            nn.Linear(self.latent_dim + condition_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

        # Move to device
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        device = torch.device(self._device)
        self._encoder = self._encoder.to(device)
        self._enc_mu = self._enc_mu.to(device)
        self._enc_logvar = self._enc_logvar.to(device)
        self._decoder = self._decoder.to(device)

        # Optimizer
        params = (
            list(self._encoder.parameters())
            + list(self._enc_mu.parameters())
            + list(self._enc_logvar.parameters())
            + list(self._decoder.parameters())
        )
        self._optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        logger.info("Built cVAE model on device=%s (latent_dim=%d)", self._device, self.latent_dim)

    def _encode(self, x: "torch.Tensor", c: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Encode input and condition to latent distribution parameters.

        Args:
            x: Flattened input signal (batch, input_dim).
            c: One-hot class condition (batch, n_classes).

        Returns:
            Tuple of (mu, logvar) tensors, each (batch, latent_dim).
        """
        torch = _require_torch()
        xc = torch.cat([x, c], dim=1)
        h = self._encoder(xc)
        return self._enc_mu(h), self._enc_logvar(h)

    def _reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        """Reparameterization trick for VAE training.

        Args:
            mu: Mean of latent distribution (batch, latent_dim).
            logvar: Log-variance of latent distribution (batch, latent_dim).

        Returns:
            Sampled latent vector (batch, latent_dim).
        """
        torch = _require_torch()
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(self, z: "torch.Tensor", c: "torch.Tensor") -> "torch.Tensor":
        """Decode latent vector and condition to reconstructed signal.

        Args:
            z: Latent vector (batch, latent_dim).
            c: One-hot class condition (batch, n_classes).

        Returns:
            Reconstructed signal (batch, input_dim).
        """
        torch = _require_torch()
        zc = torch.cat([z, c], dim=1)
        return self._decoder(zc)

    def _vae_loss(
        self,
        recon: "torch.Tensor",
        target: "torch.Tensor",
        mu: "torch.Tensor",
        logvar: "torch.Tensor",
        kl_weight: float = 1.0,
    ) -> "torch.Tensor":
        """Compute VAE loss: reconstruction + KL divergence.

        Args:
            recon: Reconstructed signal (batch, input_dim).
            target: Original signal (batch, input_dim).
            mu: Latent mean (batch, latent_dim).
            logvar: Latent log-variance (batch, latent_dim).
            kl_weight: Weight for KL divergence term (beta-VAE).

        Returns:
            Scalar loss tensor.
        """
        torch = _require_torch()
        import torch.nn.functional as F

        recon_loss = F.mse_loss(recon, target, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss

    def train(
        self,
        epochs: list[NeuralEpoch],
        n_epochs: int = 100,
        batch_size: int = 64,
        kl_weight: float = 1.0,
    ) -> list[float]:
        """Train the cVAE on real neural epochs.

        Args:
            epochs: List of NeuralEpoch training data.
            n_epochs: Number of training epochs.
            batch_size: Mini-batch size.
            kl_weight: KL divergence weight (beta-VAE).

        Returns:
            List of per-epoch training losses.

        Raises:
            ImportError: If torch is not installed.
            ValueError: If no epochs provided.
        """
        if not epochs:
            raise ValueError("Cannot train on empty epoch list")

        torch = _require_torch()

        # Infer dimensions from data
        self.n_channels = epochs[0].n_channels
        self.n_timepoints = epochs[0].n_timepoints
        self._channel_names = list(epochs[0].channels)
        self._sfreq = epochs[0].sfreq

        # Build model if not already built
        if self._encoder is None:
            self._build_model()

        device = torch.device(self._device)

        # Prepare data tensors
        X = np.stack([e.signals.flatten() for e in epochs], axis=0).astype(np.float32)
        labels = np.array([e.label for e in epochs], dtype=np.int64)

        X_tensor = torch.from_numpy(X).to(device)

        # One-hot encode labels
        C_tensor = torch.zeros(len(epochs), self.n_classes, device=device)
        for i, lbl in enumerate(labels):
            if lbl < self.n_classes:
                C_tensor[i, lbl] = 1.0

        # Training loop
        n_samples = len(epochs)
        self._train_losses = []

        for epoch_idx in range(n_epochs):
            # Shuffle
            perm = torch.randperm(n_samples, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, n_samples, batch_size):
                batch_idx = perm[batch_start : batch_start + batch_size]
                x_batch = X_tensor[batch_idx]
                c_batch = C_tensor[batch_idx]

                # Forward pass
                mu, logvar = self._encode(x_batch, c_batch)
                z = self._reparameterize(mu, logvar)
                recon = self._decode(z, c_batch)

                # Loss
                loss = self._vae_loss(recon, x_batch, mu, logvar, kl_weight)

                # Backward pass
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self._train_losses.append(avg_loss)

            if (epoch_idx + 1) % 10 == 0:
                logger.info("cVAE epoch %d/%d, loss=%.6f", epoch_idx + 1, n_epochs, avg_loss)

        self._trained = True
        logger.info("Training complete. Final loss=%.6f", self._train_losses[-1])
        return self._train_losses

    def generate(
        self,
        class_label: int,
        drift_state: np.ndarray | None = None,
        n_samples: int = 1,
    ) -> list[NeuralEpoch]:
        """Generate synthetic neural epochs for a given class.

        Args:
            class_label: Target class label (0-indexed).
            drift_state: Optional drift state vector to condition on.
                Currently unused (reserved for drift-aware generation).
            n_samples: Number of epochs to generate.

        Returns:
            List of synthetic NeuralEpoch objects.

        Raises:
            ImportError: If torch is not installed.
            RuntimeError: If model has not been trained.
        """
        if not self._trained:
            raise RuntimeError(
                "NeuralSurrogate must be trained before generating. "
                "Call .train(epochs) first."
            )

        torch = _require_torch()
        device = torch.device(self._device)

        # Condition vector: one-hot class label
        c = torch.zeros(n_samples, self.n_classes, device=device)
        if class_label < self.n_classes:
            c[:, class_label] = 1.0

        # Sample from prior
        z = torch.randn(n_samples, self.latent_dim, device=device)

        # Decode
        with torch.no_grad():
            generated = self._decode(z, c)

        # Convert to NeuralEpoch
        gen_np = generated.cpu().numpy()
        channel_names = self._channel_names or [f"Ch{i:03d}" for i in range(self.n_channels)]

        results: list[NeuralEpoch] = []
        for i in range(n_samples):
            signals = gen_np[i].reshape(self.n_channels, self.n_timepoints)
            epoch = NeuralEpoch(
                signals=signals.astype(np.float64),
                label=class_label,
                sfreq=self._sfreq,
                channels=list(channel_names),
                subject_id=0,  # synthetic
                session_id=0,
                metadata={
                    "synthetic": True,
                    "generator": "NeuralSurrogate_cVAE",
                    "drift_state": drift_state.tolist() if drift_state is not None else None,
                },
            )
            results.append(epoch)

        return results

    def save(self, path: str | Path) -> None:
        """Save trained model weights and config to disk.

        Args:
            path: File path for the saved model (.pt).

        Raises:
            RuntimeError: If model has not been trained.
        """
        if not self._trained:
            raise RuntimeError("Cannot save untrained model")

        torch = _require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": {
                "n_channels": self.n_channels,
                "n_timepoints": self.n_timepoints,
                "latent_dim": self.latent_dim,
                "n_classes": self.n_classes,
                "learning_rate": self.learning_rate,
            },
            "encoder": self._encoder.state_dict(),
            "enc_mu": self._enc_mu.state_dict(),
            "enc_logvar": self._enc_logvar.state_dict(),
            "decoder": self._decoder.state_dict(),
            "channel_names": self._channel_names,
            "sfreq": self._sfreq,
            "train_losses": self._train_losses,
        }
        torch.save(state, path)
        logger.info("Saved NeuralSurrogate to %s", path)

    def load(self, path: str | Path) -> None:
        """Load model weights and config from disk.

        Args:
            path: File path to the saved model (.pt).

        Raises:
            FileNotFoundError: If path does not exist.
            ImportError: If torch is not installed.
        """
        torch = _require_torch()
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        state = torch.load(path, map_location="cpu", weights_only=False)

        # Restore config
        config = state["config"]
        self.n_channels = config["n_channels"]
        self.n_timepoints = config["n_timepoints"]
        self.latent_dim = config["latent_dim"]
        self.n_classes = config["n_classes"]
        self.learning_rate = config["learning_rate"]

        # Build model and load weights
        self._build_model()
        device = torch.device(self._device)

        self._encoder.load_state_dict(state["encoder"])
        self._enc_mu.load_state_dict(state["enc_mu"])
        self._enc_logvar.load_state_dict(state["enc_logvar"])
        self._decoder.load_state_dict(state["decoder"])

        self._channel_names = state.get("channel_names", [])
        self._sfreq = state.get("sfreq", 250.0)
        self._train_losses = state.get("train_losses", [])
        self._trained = True

        logger.info("Loaded NeuralSurrogate from %s (device=%s)", path, self._device)
