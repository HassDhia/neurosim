"""NeuroSim - A Gymnasium platform for RL in brain-computer interfaces.

Provides tools for training RL agents to decode, adapt to, and control
brain-computer interfaces, including:

- MOABB-backed data pipeline for 36+ standardized BCI datasets
- Gymnasium RL environments (DecoderAdapt-v0, CursorControl-v0, SpellerNav-v0)
- Signal models for non-stationarity, electrode drift, and co-adaptation
- Neural signal surrogate (cVAE) for fast training
- PPO baseline agents and benchmark suite
"""

__version__ = "0.1.0"

# Register Gymnasium environments on package import
import neurosim.envs  # noqa: F401

__all__ = ["__version__"]
