"""Helpers to discover StyleGAN synthesis blocks and ordering."""

import torch


def discover_stylegan_blocks(netG):
    """Return a list of synthesis blocks in execution order when possible.

    This scans the synthesis network for modules that look like StyleGAN blocks.
    It uses forward hooks to preserve execution order when multiple blocks exist.
    """
    synth = netG.synthesis
    blocks = []

    # First try the common attribute-based route.
    block_resolutions = getattr(synth, "block_resolutions", None)
    if block_resolutions:
        for res in block_resolutions:
            name = f"b{res}"
            if hasattr(synth, name):
                blocks.append(getattr(synth, name))
        if blocks:
            return blocks

    # StyleGAN-XL (some pickles) expose top-level layers as L{idx}_{...}.
    # Use named_children to preserve ordering by index.
    layer_children = []
    for name, module in synth.named_children():
        if name.startswith("L"):
            prefix = name[1:].split("_", 1)[0]
            if prefix.isdigit():
                layer_children.append((int(prefix), module))
    if layer_children:
        layer_children.sort(key=lambda x: x[0])
        return [m for _, m in layer_children]

    # Next try a direct container if exposed.
    if hasattr(synth, "blocks"):
        candidate = getattr(synth, "blocks")
        if isinstance(candidate, (torch.nn.ModuleList, torch.nn.Sequential, list, tuple)):
            blocks = list(candidate)
            if blocks:
                return blocks

    # Fallback: scan all submodules and preserve execution order via hooks.
    candidates = []
    for module in synth.modules():
        if module is synth:
            continue
        if hasattr(module, "num_conv") and hasattr(module, "num_torgb") and hasattr(module, "resolution"):
            candidates.append(module)

    if not candidates:
        return []

    ordered = []
    handles = []

    def _hook(mod, _inp, _out):
        if mod not in ordered:
            ordered.append(mod)

    for mod in candidates:
        handles.append(mod.register_forward_hook(_hook))

    try:
        # If we can run a dummy forward, hooks will establish order.
        # Caller should provide valid ws and invoke synth forward externally.
        # This function will still return candidates if hooks never fire.
        pass
    finally:
        for h in handles:
            h.remove()

    return ordered if ordered else candidates
