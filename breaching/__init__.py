"""Initialize breaching library :>"""

# Disable optimized SDPA kernels for compatibility with older GPUs (RTX 2080 Ti, etc.)
# This prevents "derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented" errors
# Force use of the stable math SDP kernel which supports backward passes on all GPUs
import torch
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)

# # Also set CUDA backend preferences to ensure math kernel is used
# torch.set_float32_matmul_precision('highest')

from breaching import analysis
from breaching import attacks
from breaching import cases
from breaching import utils

__all__ = ["analysis", "attacks", "cases", "servers", "utils"]


import hydra

"""Construct interfaces to cfg folders for use in packaged installations:"""


def get_config(overrides=[]):
    """Return default hydra config."""
    with hydra.initialize(config_path="config", version_base="1.1"):
        cfg = hydra.compose(config_name="cfg", overrides=overrides)
        print(f"Investigating use case {cfg.case.name} with server type {cfg.case.server.name}.")
    return cfg


def get_attack_config(attack="invertinggradients", overrides=[]):
    """Return default hydra config for a given attack."""
    with hydra.initialize(config_path="config/attack", version_base="1.1"):
        cfg = hydra.compose(config_name=attack, overrides=overrides)
        print(f"Loading attack configuration {cfg.attack_type}-{cfg.type}.")
    return cfg


def get_case_config(case="1_single_image_small", overrides=[]):
    """Return default hydra config for a given attack."""
    with hydra.initialize(config_path="config/case", version_base="1.1"):
        cfg = hydra.compose(config_name=case, overrides=overrides)
        print(f"Investigating use case {cfg.name} with server type {cfg.server.name}.")
    return cfg


# Variant (other OS?):
# with hydra.initialize_config_module(config_module="breaching.config"):
#     cfg = hydra.compose(config_name="cfg", overrides=overrides)
