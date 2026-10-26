"""Helpers to discover DCGAN generator feature stages."""

import torch


def discover_dcgan_blocks(netG):
    """Return DCGAN generator modules that mark stable feature-stage boundaries.

    For the bundled DCGAN, these are the intermediate activations after each
    upsampling block and before the final image projection.
    """
    main = getattr(netG, "main", None)
    if not isinstance(main, (torch.nn.Sequential, torch.nn.ModuleList)):
        return []

    blocks = []
    modules = list(main)
    activation_types = (
        torch.nn.ReLU,
        torch.nn.LeakyReLU,
        torch.nn.ELU,
        torch.nn.GELU,
        torch.nn.SiLU,
    )

    for idx, module in enumerate(modules):
        if not isinstance(module, activation_types):
            continue

        next_module = modules[idx + 1] if idx + 1 < len(modules) else None
        if isinstance(next_module, torch.nn.Tanh):
            continue
        blocks.append(module)

    return blocks
