from dataclasses import MISSING

from isaaclab.utils import configclass

@configclass
class RslRlRleCfg:
    """Configuration for the RLE (Reinforcement Learning Environment) module."""

    input_size: int = MISSING

    weight: float = 0.0
    """The weight of the RLE module."""

    feature_size : int = 32

    num_envs : int = 4096

    tau : float = 0.005

    update_steps: int = 16
    
    num_iterations_feat_norm_init: int = 1
    
    num_steps: int = 16