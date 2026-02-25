"""MOABB dataset loader for NeuroSim.

Bridges the MOABB BCI dataset ecosystem into NeuroSim's internal NeuralEpoch
format. Supports all major motor-imagery and P300 datasets used in BCI research.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from neurosim.data.formats import NeuralEpoch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Priority datasets from the NeuroSim architecture spec.
# These cover the primary BCI paradigms with sufficient subject counts
# and session diversity for meaningful non-stationarity experiments.
# ──────────────────────────────────────────────────────────────────────
PRIORITY_DATASETS: list[str] = [
    "BNCI2014_001",  # 9 subjects, 2 sessions, 4-class MI
    "BNCI2014_004",  # 9 subjects, 5 sessions, 2-class MI
    "BNCI2015_001",  # 12 subjects, 2-3 sessions, 2-class MI
    "Cho2017",       # 52 subjects, 1 session, 2-class MI
    "Lee2019_MI",    # 54 subjects, 2 sessions, 2-class MI
    "PhysionetMI",   # 109 subjects, 1 session, 4-class MI
    "Weibo2014",     # 10 subjects, 1 session, 7-class MI
    "Zhou2016",      # 4 subjects, 3 sessions, 3-class MI
]


class MOABBLoader:
    """Loads BCI datasets from MOABB and converts to NeuralEpoch format.

    This loader wraps MOABB's dataset/paradigm API to produce a flat list
    of NeuralEpoch objects suitable for the NeuroSim simulation pipeline.

    Args:
        dataset: MOABB dataset name (must be in the registry).
        subjects: List of subject IDs to load. None loads all subjects.
        paradigm: MOABB paradigm name. Defaults to "MotorImagery".

    Example::

        loader = MOABBLoader("BNCI2014_001", subjects=[1, 2])
        epochs = loader.load()
        print(f"Loaded {len(epochs)} epochs")
    """

    # Maps dataset string names to their MOABB module paths for lazy import.
    _dataset_registry: dict[str, str] = {
        "BNCI2014_001": "moabb.datasets.BNCI2014_001",
        "BNCI2014_004": "moabb.datasets.BNCI2014_004",
        "BNCI2015_001": "moabb.datasets.BNCI2015_001",
        "Cho2017": "moabb.datasets.Cho2017",
        "Lee2019_MI": "moabb.datasets.Lee2019_MI",
        "PhysionetMI": "moabb.datasets.PhysionetMI",
        "Weibo2014": "moabb.datasets.Weibo2014",
        "Zhou2016": "moabb.datasets.Zhou2016",
        "AlexMI": "moabb.datasets.AlexMI",
        "BNCI2014_002": "moabb.datasets.BNCI2014_002",
        "BNCI2015_004": "moabb.datasets.BNCI2015_004",
        "Schirrmeister2017": "moabb.datasets.Schirrmeister2017",
    }

    def __init__(
        self,
        dataset: str = "BNCI2014_001",
        subjects: list[int] | None = None,
        paradigm: str = "MotorImagery",
    ) -> None:
        if dataset not in self._dataset_registry:
            available = ", ".join(sorted(self._dataset_registry.keys()))
            raise ValueError(
                f"Unknown dataset '{dataset}'. Available: {available}"
            )
        self.dataset_name = dataset
        self.subjects = subjects
        self.paradigm_name = paradigm
        self._dataset_obj: Any = None
        self._paradigm_obj: Any = None

    def _resolve_moabb_objects(self) -> None:
        """Lazily resolve MOABB dataset and paradigm objects.

        Raises:
            ImportError: If moabb is not installed.
        """
        try:
            import moabb
            from moabb import datasets as moabb_datasets
            from moabb import paradigms as moabb_paradigms
        except ImportError as exc:
            raise ImportError(
                "MOABB is required for dataset loading but is not installed.\n"
                "Install it with: pip install moabb\n"
                "Or for the full NeuroSim stack: pip install neurosim[data]"
            ) from exc

        # Resolve dataset class from MOABB
        dataset_cls_name = self.dataset_name
        if not hasattr(moabb_datasets, dataset_cls_name):
            raise ValueError(
                f"MOABB does not expose dataset class '{dataset_cls_name}'. "
                f"Check MOABB version compatibility."
            )
        dataset_cls = getattr(moabb_datasets, dataset_cls_name)
        self._dataset_obj = dataset_cls()

        # Override subject list if specified
        if self.subjects is not None:
            self._dataset_obj.subject_list = self.subjects

        # Resolve paradigm
        paradigm_map: dict[str, str] = {
            "MotorImagery": "MotorImagery",
            "LeftRightImagery": "LeftRightImagery",
            "FilterBankMotorImagery": "FilterBankMotorImagery",
            "P300": "P300",
            "SSVEP": "SSVEP",
        }
        paradigm_cls_name = paradigm_map.get(self.paradigm_name, self.paradigm_name)
        if not hasattr(moabb_paradigms, paradigm_cls_name):
            raise ValueError(
                f"Unknown paradigm '{self.paradigm_name}'. "
                f"Available: {list(paradigm_map.keys())}"
            )
        paradigm_cls = getattr(moabb_paradigms, paradigm_cls_name)
        self._paradigm_obj = paradigm_cls()

    def load(self) -> list[NeuralEpoch]:
        """Load dataset via MOABB and convert to NeuralEpoch list.

        Returns:
            List of NeuralEpoch objects, one per trial/epoch.

        Raises:
            ImportError: If moabb is not installed.
            ValueError: If dataset or paradigm is invalid.
        """
        self._resolve_moabb_objects()

        logger.info(
            "Loading dataset=%s paradigm=%s subjects=%s",
            self.dataset_name,
            self.paradigm_name,
            self.subjects or "all",
        )

        # MOABB get_data returns (X, labels, meta) arrays
        X, labels, meta = self._paradigm_obj.get_data(
            dataset=self._dataset_obj
        )

        # X shape: (n_epochs, n_channels, n_timepoints)
        # labels: (n_epochs,) string labels
        # meta: DataFrame with subject, session, run info

        # Build label encoder (string -> int)
        unique_labels = sorted(set(labels))
        label_to_int: dict[str, int] = {
            lbl: idx for idx, lbl in enumerate(unique_labels)
        }

        # Extract channel info from dataset
        sfreq = self._paradigm_obj.resample if hasattr(self._paradigm_obj, "resample") and self._paradigm_obj.resample else 250.0
        channel_names = self._get_channel_names(X.shape[1])

        epochs: list[NeuralEpoch] = []
        for i in range(X.shape[0]):
            subject_id = int(meta.iloc[i].get("subject", 0))
            session_id = int(
                str(meta.iloc[i].get("session", "0")).replace("session_", "")
            ) if "session" in meta.columns else 0

            epoch = NeuralEpoch(
                signals=X[i].astype(np.float64),
                label=label_to_int[labels[i]],
                sfreq=sfreq,
                channels=channel_names,
                subject_id=subject_id,
                session_id=session_id,
                metadata={
                    "original_label": labels[i],
                    "label_map": label_to_int,
                    "dataset": self.dataset_name,
                },
            )
            epochs.append(epoch)

        logger.info(
            "Loaded %d epochs (%d subjects, %d classes)",
            len(epochs),
            len(set(e.subject_id for e in epochs)),
            len(unique_labels),
        )
        return epochs

    def _get_channel_names(self, n_channels: int) -> list[str]:
        """Extract or generate channel names.

        Args:
            n_channels: Number of channels in the data.

        Returns:
            List of channel name strings.
        """
        if self._dataset_obj is not None:
            try:
                # Try to get channel info from the dataset
                raw = self._dataset_obj.get_data(
                    subjects=[self._dataset_obj.subject_list[0]]
                )
                if isinstance(raw, dict):
                    for subj_data in raw.values():
                        for sess_data in subj_data.values():
                            for run_data in sess_data.values():
                                if hasattr(run_data, "ch_names"):
                                    names = [
                                        ch for ch in run_data.ch_names
                                        if ch not in ("stim", "STI 014", "Status")
                                    ]
                                    if len(names) >= n_channels:
                                        return names[:n_channels]
            except Exception:
                pass

        # Fallback: generate generic channel names
        return [f"Ch{i:03d}" for i in range(n_channels)]

    def available_subjects(self) -> list[int]:
        """Return available subject IDs for the configured dataset.

        Returns:
            List of subject IDs.
        """
        self._resolve_moabb_objects()
        return list(self._dataset_obj.subject_list)

    def __repr__(self) -> str:
        return (
            f"MOABBLoader(dataset={self.dataset_name!r}, "
            f"subjects={self.subjects!r}, paradigm={self.paradigm_name!r})"
        )
