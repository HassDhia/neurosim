"""Tests for MOABBLoader — all MOABB dependencies are mocked."""
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from neurosim.data.loader import MOABBLoader, PRIORITY_DATASETS


class TestMOABBLoaderInit:
    def test_invalid_dataset_raises(self):
        """Unknown dataset name raises ValueError immediately."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            MOABBLoader("FakeDataset")

    def test_valid_dataset_names(self):
        """Every PRIORITY_DATASETS entry is in the loader registry."""
        for ds_name in PRIORITY_DATASETS:
            loader = MOABBLoader(ds_name)
            assert loader.dataset_name == ds_name

    def test_repr(self):
        """repr returns an informative string with dataset/paradigm info."""
        loader = MOABBLoader("BNCI2014_001", subjects=[1, 2])
        r = repr(loader)
        assert "BNCI2014_001" in r
        assert "MotorImagery" in r
        assert "[1, 2]" in r


class TestMOABBLoaderLoad:
    """Tests that mock MOABB imports to verify the load pipeline."""

    def _build_mock_moabb(self, n_epochs=10, n_channels=22, n_timepoints=256):
        """Create mock moabb module hierarchy."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_epochs, n_channels, n_timepoints))
        labels = np.array(["left_hand", "right_hand"] * (n_epochs // 2))
        meta = pd.DataFrame({
            "subject": [1] * n_epochs,
            "session": ["session_0"] * n_epochs,
            "run": ["run_0"] * n_epochs,
        })

        mock_paradigm_instance = MagicMock()
        mock_paradigm_instance.get_data.return_value = (X, labels, meta)
        mock_paradigm_instance.resample = None

        mock_paradigm_cls = MagicMock(return_value=mock_paradigm_instance)

        mock_dataset_instance = MagicMock()
        mock_dataset_instance.subject_list = [1]

        mock_dataset_cls = MagicMock(return_value=mock_dataset_instance)

        # Build module mocks
        mock_moabb = MagicMock()
        mock_datasets = MagicMock()
        mock_paradigms = MagicMock()

        # Set the dataset class on the datasets module
        setattr(mock_datasets, "BNCI2014_001", mock_dataset_cls)

        # Set paradigm class on the paradigms module
        setattr(mock_paradigms, "MotorImagery", mock_paradigm_cls)

        return mock_moabb, mock_datasets, mock_paradigms, X, labels

    def test_load_converts_to_neural_epochs(self):
        """Mock MOABB, verify NeuralEpoch list with correct shapes/labels."""
        mock_moabb, mock_datasets, mock_paradigms, X, labels = self._build_mock_moabb(
            n_epochs=10, n_channels=22, n_timepoints=256
        )
        loader = MOABBLoader("BNCI2014_001")

        with patch.dict(sys.modules, {
            "moabb": mock_moabb,
            "moabb.datasets": mock_datasets,
            "moabb.paradigms": mock_paradigms,
        }):
            # Patch the import inside _resolve_moabb_objects
            mock_moabb.datasets = mock_datasets
            mock_moabb.paradigms = mock_paradigms

            # Directly set the resolved objects to bypass import machinery
            loader._dataset_obj = mock_datasets.BNCI2014_001()
            loader._paradigm_obj = mock_paradigms.MotorImagery()

            # Call load's data processing logic
            X_data, labels_data, meta = loader._paradigm_obj.get_data(
                dataset=loader._dataset_obj
            )

            # Verify MOABB returned expected shapes
            assert X_data.shape == (10, 22, 256)

            # Now call actual load with patched imports
            loader._dataset_obj = None
            loader._paradigm_obj = None

            with patch.object(loader, "_resolve_moabb_objects") as mock_resolve:
                def setup_objects():
                    loader._dataset_obj = mock_datasets.BNCI2014_001()
                    loader._paradigm_obj = mock_paradigms.MotorImagery()

                mock_resolve.side_effect = setup_objects
                with patch.object(loader, "_get_channel_names",
                                  return_value=[f"Ch{i:03d}" for i in range(22)]):
                    epochs = loader.load()

        assert len(epochs) == 10
        assert epochs[0].n_channels == 22
        assert epochs[0].n_timepoints == 256
        assert all(e.label in (0, 1) for e in epochs)

    def test_label_encoding(self):
        """String labels are encoded as 0-indexed sorted integers."""
        mock_moabb, mock_datasets, mock_paradigms, X, labels = self._build_mock_moabb(
            n_epochs=6, n_channels=8, n_timepoints=100
        )

        # Override labels to have 3 classes
        rng = np.random.default_rng(99)
        X_small = rng.standard_normal((6, 8, 100))
        three_labels = np.array(["right_hand", "feet", "left_hand", "feet", "left_hand", "right_hand"])
        meta = pd.DataFrame({
            "subject": [1] * 6,
            "session": ["session_0"] * 6,
        })
        mock_paradigm_instance = MagicMock()
        mock_paradigm_instance.get_data.return_value = (X_small, three_labels, meta)
        mock_paradigm_instance.resample = None
        mock_paradigms.MotorImagery.return_value = mock_paradigm_instance

        loader = MOABBLoader("BNCI2014_001")

        with patch.object(loader, "_resolve_moabb_objects") as mock_resolve:
            def setup():
                loader._dataset_obj = mock_datasets.BNCI2014_001()
                loader._paradigm_obj = mock_paradigms.MotorImagery()

            mock_resolve.side_effect = setup
            with patch.object(loader, "_get_channel_names",
                              return_value=[f"Ch{i:03d}" for i in range(8)]):
                epochs = loader.load()

        # Sorted unique labels: feet=0, left_hand=1, right_hand=2
        label_set = {e.label for e in epochs}
        assert label_set == {0, 1, 2}
        # Verify mapping via metadata
        for e in epochs:
            orig = e.metadata["original_label"]
            expected_map = {"feet": 0, "left_hand": 1, "right_hand": 2}
            assert e.label == expected_map[orig]


class TestMOABBNotInstalled:
    def test_moabb_not_installed_raises(self):
        """Clear ImportError message when moabb is missing."""
        loader = MOABBLoader("BNCI2014_001")
        with patch.dict(sys.modules, {"moabb": None}):
            with pytest.raises(ImportError, match="MOABB is required"):
                loader._resolve_moabb_objects()
