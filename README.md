# NeuroSim

**A Gymnasium Platform for Reinforcement Learning in Brain-Computer Interfaces**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Tests](https://img.shields.io/badge/tests-158%20passing-brightgreen.svg)
[![PyPI version](https://img.shields.io/pypi/v/neurosim.svg)](https://pypi.org/project/neurosim/)

---

NeuroSim is a Gymnasium-compatible reinforcement learning environment suite for brain-computer interfaces. It provides three environments modeling motor imagery decoding, intracortical cursor control, and P300 speller stimulus selection. The package includes pluggable signal models (electrode drift, fatigue, co-adaptation, noise), a conditional VAE neural surrogate, CSP+LDA classical baseline, PPO RL baseline, and a five-tier benchmark suite for reproducible evaluation across drift conditions and subject variability.

## Installation

```bash
pip install neurosim              # Core (numpy, scipy, gymnasium)
pip install neurosim[train]       # + SB3, PyTorch for RL training
pip install neurosim[data]        # + MNE, MOABB for real EEG data
pip install neurosim[surrogate]   # + PyTorch for neural surrogate
pip install neurosim[all]         # Everything
```

Development install:

```bash
git clone https://github.com/HassDhia/neurosim.git
cd neurosim
pip install -e ".[all]"
```

## Quick Start

```python
import gymnasium as gym
import neurosim

env = gym.make("neurosim/DecoderAdapt-v0")
obs, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

## Environments

| Environment | Paradigm | Observation | Action | Key Challenge |
|---|---|---|---|---|
| `DecoderAdapt-v0` | Motor imagery | 55-dim (neural + confidence + drift) | Dict: classify, recalibrate, adapt | When to recalibrate under electrode drift |
| `CursorControl-v0` | Intracortical | 96x5 neural state + cursor + target | Box(2) velocity | Continuous control through neural noise |
| `SpellerNav-v0` | P300 speller | 36-posterior + ERP history | Dict: stimulus group, commit | Balance evidence vs communication speed |

## Signal Models

NeuroSim implements pluggable signal degradation models that simulate real-world BCI non-stationarity:

- **Electrode drift** -- impedance degradation over session time, modeled as monotonic SNR reduction per channel
- **Fatigue drift** -- exponential saturation of motor imagery signal quality with sustained use
- **Feature shift** -- sudden and continuous distributional shifts in the neural feature space
- **Co-adaptation** -- bidirectional user-decoder coupling where the user's neural patterns change in response to decoder behavior
- **Noise injection** -- pink (1/f) noise, EMG artifact, blink artifact, and 50/60 Hz line noise at configurable intensities

## Benchmark Tiers

| Tier | Condition | Drift Sources |
|---|---|---|
| T1 | Stationary -- no drift, low noise | None |
| T2 | Mild -- electrode drift, medium noise | Electrode |
| T3 | Full -- all drift, medium noise | Electrode, fatigue, feature, co-adaptation |
| T4 | Cross-subject -- multi-subject, high noise | All |
| T5 | Adversarial -- sudden shifts, high noise | All + sudden |

## Training Agents

Train a PPO agent and run the benchmark suite from the command line:

```bash
neurosim-train --env DecoderAdapt-v0 --steps 100000
neurosim-benchmark --tier T1 --episodes 50
```

Or use Stable-Baselines3 directly:

```python
from stable_baselines3 import PPO
import gymnasium as gym
import neurosim

env = gym.make("neurosim/DecoderAdapt-v0")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_decoder_adapt")
```

## Citation

If you use NeuroSim in your research, please cite:

```bibtex
@software{neurosim2026,
  author = {Dhia, Hass},
  title = {NeuroSim: A Gymnasium Platform for RL in Brain-Computer Interfaces},
  year = {2026},
  url = {https://github.com/HassDhia/neurosim},
}
```

## License

MIT -- see [LICENSE](LICENSE) for details.
