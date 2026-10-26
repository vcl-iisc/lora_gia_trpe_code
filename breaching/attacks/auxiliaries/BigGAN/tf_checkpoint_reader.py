# coding: utf-8
"""
Minimal TF checkpoint reader without TensorFlow.

This uses TensorStore's TensorFlow checkpoint driver to read variables directly
from a TF checkpoint prefix (e.g., ".../variables/variables").

Requires: `pip install tensorstore`
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
from typing import Iterable, Dict


def _require_tensorstore():
    try:
        import tensorstore as ts  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Reading TF checkpoints without TensorFlow requires `tensorstore` "
            "(pip install tensorstore)."
        ) from exc
    return ts


def resolve_checkpoint_prefix(path: str) -> str:
    """
    Resolve a TF checkpoint prefix from a directory or prefix.

    If `path` is a directory and contains a `variables/variables.*` checkpoint,
    return that prefix; otherwise return `path` unchanged.
    """
    if os.path.isdir(path):
        candidate = os.path.join(path, "variables", "variables")
        if os.path.exists(candidate + ".index"):
            return candidate
    return path


def load_variable(checkpoint_prefix: str, name: str):
    """
    Load a single variable array from a TF checkpoint prefix.
    """
    ts = _require_tensorstore()
    ckpt = resolve_checkpoint_prefix(checkpoint_prefix)
    spec = {"driver": "tensorflow", "path": ckpt, "variable": name}
    try:
        tensor = ts.open(spec, open=True).result()
    except ValueError as exc:
        raise RuntimeError(
            "TensorStore was installed without the `tensorflow` driver. "
            "Install a TensorStore build that includes TF checkpoint support, "
            "or use TensorFlow+TF Hub to extract BN stats, or provide "
            "`batch_norm_stats_path`."
        ) from exc
    return tensor.read().result()


def load_variables(checkpoint_prefix: str, names: Iterable[str]) -> Dict[str, object]:
    """
    Load multiple variables from a TF checkpoint prefix.
    """
    return {name: load_variable(checkpoint_prefix, name) for name in names}
