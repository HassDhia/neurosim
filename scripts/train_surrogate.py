#!/usr/bin/env python3
"""Train the cVAE neural surrogate on synthetic data.

Usage::
    python scripts/train_surrogate.py
    python scripts/train_surrogate.py --n-channels 22 --n-timepoints 250 --n-classes 4 --n-epochs 200
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the project src is importable when running from repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neurosim.data.formats import NeuralEpoch
from neurosim.models.surrogate import NeuralSurrogate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the NeuralSurrogate cVAE on synthetic EEG data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-channels", type=int, default=22,
        help="Number of EEG channels.",
    )
    parser.add_argument(
        "--n-timepoints", type=int, default=250,
        help="Number of time samples per epoch.",
    )
    parser.add_argument(
        "--n-classes", type=int, default=4,
        help="Number of BCI output classes.",
    )
    parser.add_argument(
        "--n-per-class", type=int, default=50,
        help="Number of training epochs to generate per class.",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=100,
        help="Number of training epochs for the cVAE.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size for training.",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=32,
        help="Dimensionality of the VAE latent space.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Adam optimizer learning rate.",
    )
    parser.add_argument(
        "--kl-weight", type=float, default=1.0,
        help="KL divergence weight (beta-VAE).",
    )
    parser.add_argument(
        "--sfreq", type=float, default=250.0,
        help="Simulated sampling frequency in Hz.",
    )
    parser.add_argument(
        "--noise-scale", type=float, default=0.5,
        help="Standard deviation of additive Gaussian noise.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output", type=str, default="results/models/surrogate.pt",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--gen-samples", type=int, default=5,
        help="Number of samples per class to generate after training.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser.parse_args()


def generate_synthetic_epochs(
    n_channels: int,
    n_timepoints: int,
    n_classes: int,
    n_per_class: int,
    sfreq: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> list[NeuralEpoch]:
    """Generate synthetic NeuralEpoch training data with class-conditioned means.

    Each class k has a deterministic mean offset applied to every k-th channel
    (starting from channel k, stepping by n_classes). This gives the cVAE a
    learnable signal to separate classes.

    Args:
        n_channels: Number of EEG channels.
        n_timepoints: Number of time samples per epoch.
        n_classes: Number of BCI output classes.
        n_per_class: Number of epochs to generate per class.
        sfreq: Sampling frequency in Hz.
        noise_scale: Standard deviation of additive Gaussian noise.
        rng: NumPy random generator for reproducibility.

    Returns:
        List of NeuralEpoch objects (n_classes * n_per_class total).
    """
    channel_names = [f"Ch{i:03d}" for i in range(n_channels)]
    epochs: list[NeuralEpoch] = []

    for class_k in range(n_classes):
        # Deterministic class offset: channels k, k+n_classes, k+2*n_classes, ... get +2.0
        class_offset = np.zeros(n_channels)
        class_offset[class_k::n_classes] = 2.0

        for sample_i in range(n_per_class):
            noise = rng.normal(loc=0.0, scale=noise_scale, size=(n_channels, n_timepoints))
            signals = class_offset[:, np.newaxis] + noise

            epoch = NeuralEpoch(
                signals=signals,
                label=class_k,
                sfreq=sfreq,
                channels=list(channel_names),
                subject_id=0,
                session_id=0,
                metadata={"synthetic_source": "train_surrogate.py"},
            )
            epochs.append(epoch)

    return epochs


def print_loss_summary(losses: list[float]) -> None:
    """Print a compact summary of the training loss curve."""
    if not losses:
        print("  (no losses recorded)")
        return

    first = losses[0]
    final = losses[-1]
    min_loss = min(losses)
    min_idx = losses.index(min_loss)
    reduction_pct = ((first - final) / first) * 100 if first > 0 else 0.0

    print(f"  First epoch loss : {first:.6f}")
    print(f"  Min loss         : {min_loss:.6f}  (epoch {min_idx + 1})")
    print(f"  Final epoch loss : {final:.6f}")
    print(f"  Reduction        : {reduction_pct:.1f}%")

    # Print a 10-point sparkline of the loss curve
    n = len(losses)
    if n >= 10:
        indices = [int(i * (n - 1) / 9) for i in range(10)]
        sampled = [losses[i] for i in indices]
        lo, hi = min(sampled), max(sampled)
        sparkline_chars = " _.,:-=!#"
        if hi > lo:
            sparkline = "".join(
                sparkline_chars[int((v - lo) / (hi - lo) * (len(sparkline_chars) - 1))]
                for v in sampled
            )
        else:
            sparkline = "_" * 10
        print(f"  Loss trend       : [{sparkline}]")


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()

    # Logging setup
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("train_surrogate")

    # Seed
    rng = np.random.default_rng(args.seed)

    total_epochs = args.n_classes * args.n_per_class
    print("=" * 60)
    print("NeuroSim cVAE Surrogate Training")
    print("=" * 60)
    print(f"  Channels     : {args.n_channels}")
    print(f"  Timepoints   : {args.n_timepoints}")
    print(f"  Classes      : {args.n_classes}")
    print(f"  Per class    : {args.n_per_class}")
    print(f"  Total epochs : {total_epochs}")
    print(f"  Latent dim   : {args.latent_dim}")
    print(f"  Train epochs : {args.n_epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  LR           : {args.learning_rate}")
    print(f"  KL weight    : {args.kl_weight}")
    print(f"  Noise scale  : {args.noise_scale}")
    print(f"  Seed         : {args.seed}")
    print(f"  Output       : {args.output}")
    print()

    # --- Step 1: Generate synthetic training data ---
    print("[1/4] Generating synthetic training data...")
    train_data = generate_synthetic_epochs(
        n_channels=args.n_channels,
        n_timepoints=args.n_timepoints,
        n_classes=args.n_classes,
        n_per_class=args.n_per_class,
        sfreq=args.sfreq,
        noise_scale=args.noise_scale,
        rng=rng,
    )
    print(f"      Generated {len(train_data)} epochs "
          f"({args.n_channels}ch x {args.n_timepoints}tp)")
    print()

    # --- Step 2: Train the NeuralSurrogate ---
    print("[2/4] Training NeuralSurrogate cVAE...")
    surrogate = NeuralSurrogate(
        n_channels=args.n_channels,
        n_timepoints=args.n_timepoints,
        latent_dim=args.latent_dim,
        n_classes=args.n_classes,
        learning_rate=args.learning_rate,
    )

    t_start = time.perf_counter()
    losses = surrogate.train(
        train_data,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        kl_weight=args.kl_weight,
    )
    t_elapsed = time.perf_counter() - t_start

    print(f"      Training complete in {t_elapsed:.1f}s")
    print()
    print("      Loss curve summary:")
    print_loss_summary(losses)
    print()

    # --- Step 3: Generate sample synthetic epochs ---
    print(f"[3/4] Generating {args.gen_samples} samples per class...")
    for class_k in range(args.n_classes):
        generated = surrogate.generate(class_label=class_k, n_samples=args.gen_samples)
        signals_stack = np.stack([e.signals for e in generated])
        mean_val = signals_stack.mean()
        std_val = signals_stack.std()
        print(f"      Class {class_k}: {len(generated)} samples, "
              f"mean={mean_val:+.4f}, std={std_val:.4f}, "
              f"shape={generated[0].signals.shape}")
    print()

    # --- Step 4: Save the trained model ---
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    print(f"[4/4] Saving model to {output_path}...")
    surrogate.save(output_path)
    file_size_kb = output_path.stat().st_size / 1024
    print(f"      Saved ({file_size_kb:.1f} KB)")
    print()

    print("=" * 60)
    print("Done. Model ready for use with NeuralSurrogate.load()")
    print("=" * 60)


if __name__ == "__main__":
    main()
