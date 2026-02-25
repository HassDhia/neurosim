"""Benchmark metrics for BCI decoder evaluation.

Standard metrics used across all NeuroSim benchmark tiers:
- Classification accuracy
- Information Transfer Rate (ITR)
- Reach time (cursor control)
- Path efficiency
- Adaptation efficiency
"""
from __future__ import annotations

import numpy as np


def classification_accuracy(
    predictions: np.ndarray | list[int],
    labels: np.ndarray | list[int],
) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Predicted class labels.
        labels: Ground-truth class labels.

    Returns:
        Accuracy as a float in [0, 1].

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}"
        )
    if len(predictions) == 0:
        raise ValueError("Cannot compute accuracy on empty arrays")
    return float(np.mean(predictions == labels))


def information_transfer_rate(
    n_classes: int,
    accuracy: float,
    trial_time: float,
) -> float:
    """Compute Information Transfer Rate (ITR) in bits/minute.

    Uses Wolpaw's ITR formula:
        B = log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))
        ITR = B * (60 / T)

    Args:
        n_classes: Number of classes (N >= 2).
        accuracy: Classification accuracy P in (0, 1].
        trial_time: Average trial duration in seconds (T > 0).

    Returns:
        ITR in bits per minute.

    Raises:
        ValueError: If parameters are out of valid range.
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")
    if not 0.0 < accuracy <= 1.0:
        raise ValueError(f"accuracy must be in (0, 1], got {accuracy}")
    if trial_time <= 0:
        raise ValueError(f"trial_time must be > 0, got {trial_time}")

    # Clamp accuracy to avoid log(0)
    p = min(accuracy, 0.9999)

    if n_classes == 1:
        bits_per_trial = 0.0
    else:
        bits_per_trial = (
            np.log2(n_classes)
            + p * np.log2(p)
            + (1 - p) * np.log2((1 - p) / (n_classes - 1))
        )

    # Ensure non-negative (can go negative if accuracy < 1/N)
    bits_per_trial = max(bits_per_trial, 0.0)

    return float(bits_per_trial * (60.0 / trial_time))


def reach_time(
    cursor_positions: np.ndarray,
    target_position: np.ndarray,
    threshold: float = 0.05,
) -> float:
    """Compute time to reach target in cursor control task.

    Finds the first timestep where the cursor is within threshold
    distance of the target.

    Args:
        cursor_positions: Array of cursor positions (n_steps, 2).
        target_position: Target position (2,).
        threshold: Distance threshold for "reached" (normalized units).

    Returns:
        Number of steps to reach target. Returns float('inf') if
        target was never reached.
    """
    cursor_positions = np.asarray(cursor_positions)
    target_position = np.asarray(target_position)

    distances = np.linalg.norm(cursor_positions - target_position, axis=1)
    reached_indices = np.where(distances <= threshold)[0]

    if len(reached_indices) == 0:
        return float("inf")
    return float(reached_indices[0])


def path_efficiency(
    actual_path: np.ndarray,
    optimal_distance: float,
) -> float:
    """Compute path efficiency (optimal distance / actual path length).

    Args:
        actual_path: Sequence of positions (n_steps, 2).
        optimal_distance: Straight-line distance from start to target.

    Returns:
        Efficiency ratio in (0, 1]. Returns 0 if actual path has
        zero length.

    Raises:
        ValueError: If optimal_distance is non-positive.
    """
    if optimal_distance <= 0:
        raise ValueError(f"optimal_distance must be > 0, got {optimal_distance}")

    actual_path = np.asarray(actual_path)
    if len(actual_path) < 2:
        return 0.0

    # Sum of step-wise distances
    step_distances = np.linalg.norm(np.diff(actual_path, axis=0), axis=1)
    actual_length = float(np.sum(step_distances))

    if actual_length == 0:
        return 0.0

    return float(min(optimal_distance / actual_length, 1.0))


def adaptation_efficiency(
    pre_drift_accuracy: float,
    post_drift_accuracy: float,
) -> float:
    """Compute how well the agent recovers after drift onset.

    Ratio of post-drift to pre-drift accuracy. Values close to 1.0
    indicate robust adaptation; values << 1.0 indicate poor recovery.

    Args:
        pre_drift_accuracy: Accuracy before drift onset.
        post_drift_accuracy: Accuracy after drift onset (steady-state).

    Returns:
        Adaptation efficiency ratio in [0, inf).

    Raises:
        ValueError: If pre_drift_accuracy is zero or negative.
    """
    if pre_drift_accuracy <= 0:
        raise ValueError(
            f"pre_drift_accuracy must be > 0, got {pre_drift_accuracy}"
        )
    return float(post_drift_accuracy / pre_drift_accuracy)
