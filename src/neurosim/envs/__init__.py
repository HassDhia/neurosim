"""Gymnasium environments for brain-computer interface RL."""

from gymnasium.envs.registration import register
from neurosim.envs.wrappers import wrap_for_sb3

register(
    id="neurosim/DecoderAdapt-v0",
    entry_point="neurosim.envs.decoder_adapt:DecoderAdaptEnv",
)

register(
    id="neurosim/CursorControl-v0",
    entry_point="neurosim.envs.cursor_control:CursorControlEnv",
)

register(
    id="neurosim/SpellerNav-v0",
    entry_point="neurosim.envs.speller_nav:SpellerNavEnv",
)
