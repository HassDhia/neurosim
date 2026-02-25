"""Classical BCI baseline: Common Spatial Patterns + Linear Discriminant Analysis.

Implements CSP feature extraction and a numpy-only LDA classifier,
avoiding sklearn as a dependency. This represents the traditional
BCI decoding pipeline that RL agents should outperform.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg

from neurosim.data.formats import NeuralEpoch


class CSPLDABaseline:
    """CSP + LDA classifier for motor imagery BCI.

    Uses Common Spatial Patterns for spatial filtering and a
    numpy-implemented Linear Discriminant Analysis for classification.
    Supports binary classification (2 classes).

    Args:
        n_components: Number of CSP components to keep (top + bottom).
            Must be even; half from each class's discriminative direction.
    """

    def __init__(self, n_components: int = 6) -> None:
        if n_components % 2 != 0:
            raise ValueError(f"n_components must be even, got {n_components}")
        self.n_components = n_components
        self._csp_filters: np.ndarray | None = None
        self._lda_w: np.ndarray | None = None
        self._lda_b: float = 0.0
        self._classes: np.ndarray | None = None
        self._is_fitted: bool = False

    def _compute_csp(
        self, epochs_class0: list[np.ndarray], epochs_class1: list[np.ndarray]
    ) -> np.ndarray:
        """Compute CSP spatial filters from two classes of epochs.

        Args:
            epochs_class0: List of (n_channels, n_timepoints) arrays for class 0.
            epochs_class1: List of (n_channels, n_timepoints) arrays for class 1.

        Returns:
            CSP filter matrix (n_components, n_channels).
        """
        # Compute average covariance for each class
        cov0 = np.mean(
            [np.cov(e) for e in epochs_class0], axis=0
        )
        cov1 = np.mean(
            [np.cov(e) for e in epochs_class1], axis=0
        )

        # Composite covariance
        cov_composite = cov0 + cov1

        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = linalg.eigh(cov0, cov_composite)

        # Sort by eigenvalue (descending for class 0 discrimination)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Take top and bottom components
        n_half = self.n_components // 2
        selected = np.concatenate(
            [eigenvectors[:, :n_half], eigenvectors[:, -n_half:]], axis=1
        )
        return selected.T  # (n_components, n_channels)

    def _extract_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract log-variance CSP features from a single epoch.

        Args:
            signal: Neural signal (n_channels, n_timepoints).

        Returns:
            Feature vector of shape (n_components,).
        """
        if self._csp_filters is None:
            raise RuntimeError("CSP filters not fitted. Call fit() first.")
        # Apply spatial filters
        filtered = self._csp_filters @ signal  # (n_components, n_timepoints)
        # Log-variance features
        variances = np.var(filtered, axis=1)
        # Normalize and log-transform
        variances = variances / np.sum(variances)
        return np.log(variances + 1e-10)

    def _fit_lda(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit LDA classifier using numpy (no sklearn).

        Implements Fisher's Linear Discriminant for binary classification.

        Args:
            features: Feature matrix (n_samples, n_features).
            labels: Binary class labels (n_samples,).
        """
        classes = np.unique(labels)
        if len(classes) != 2:
            raise ValueError(f"LDA requires exactly 2 classes, got {len(classes)}")
        self._classes = classes

        # Class means
        mask0 = labels == classes[0]
        mask1 = labels == classes[1]
        mu0 = np.mean(features[mask0], axis=0)
        mu1 = np.mean(features[mask1], axis=0)

        # Within-class scatter matrix
        s0 = (features[mask0] - mu0).T @ (features[mask0] - mu0)
        s1 = (features[mask1] - mu1).T @ (features[mask1] - mu1)
        s_w = s0 + s1

        # Add regularization for numerical stability
        s_w += np.eye(s_w.shape[0]) * 1e-6

        # Fisher's discriminant direction
        self._lda_w = np.linalg.solve(s_w, mu0 - mu1)

        # Decision boundary (project class means)
        proj_mu0 = self._lda_w @ mu0
        proj_mu1 = self._lda_w @ mu1
        self._lda_b = -(proj_mu0 + proj_mu1) / 2.0

    def fit(self, epochs: list[NeuralEpoch]) -> CSPLDABaseline:
        """Fit CSP spatial filters and LDA classifier on training epochs.

        Args:
            epochs: List of labeled NeuralEpoch instances (binary labels).

        Returns:
            self for method chaining.
        """
        labels = np.array([e.label for e in epochs])
        classes = np.unique(labels)
        if len(classes) != 2:
            raise ValueError(
                f"CSP+LDA requires exactly 2 classes, got {len(classes)}: {classes}"
            )

        # Split epochs by class
        epochs_c0 = [e.signals for e in epochs if e.label == classes[0]]
        epochs_c1 = [e.signals for e in epochs if e.label == classes[1]]

        # Fit CSP filters
        self._csp_filters = self._compute_csp(epochs_c0, epochs_c1)

        # Extract features for all epochs
        features = np.array([self._extract_features(e.signals) for e in epochs])

        # Fit LDA
        self._fit_lda(features, labels)
        self._is_fitted = True
        return self

    def predict(self, epoch: NeuralEpoch) -> int:
        """Predict class label for a single epoch.

        Args:
            epoch: A NeuralEpoch to classify.

        Returns:
            Predicted class label (integer).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._lda_w is None or self._classes is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        features = self._extract_features(epoch.signals)
        projection = self._lda_w @ features + self._lda_b
        class_idx = 0 if projection > 0 else 1
        return int(self._classes[class_idx])

    def evaluate(self, test_epochs: list[NeuralEpoch]) -> dict[str, Any]:
        """Evaluate on test epochs and return accuracy + confusion matrix.

        Args:
            test_epochs: List of labeled NeuralEpoch instances.

        Returns:
            Dict with 'accuracy' (float) and 'confusion_matrix' (2x2 ndarray).
        """
        if self._classes is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        predictions = [self.predict(e) for e in test_epochs]
        labels = [e.label for e in test_epochs]

        correct = sum(p == l for p, l in zip(predictions, labels))
        accuracy = correct / len(labels) if labels else 0.0

        # Build confusion matrix (2x2)
        n_classes = len(self._classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        class_to_idx = {c: i for i, c in enumerate(self._classes)}
        for pred, true in zip(predictions, labels):
            if pred in class_to_idx and true in class_to_idx:
                cm[class_to_idx[true], class_to_idx[pred]] += 1

        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
        }
