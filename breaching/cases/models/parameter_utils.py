"""Helpers for selecting shared/trainable parameters."""


def shared_parameters(model, mode="trainable"):
    """Return the parameter list exchanged between server and users.

    Models can override this by implementing a `shared_parameters` method.
    """
    if hasattr(model, "shared_parameters"):
        params = list(model.shared_parameters())
    else:
        params = list(model.parameters())

    if mode == "trainable":
        trainable = [p for p in params if p.requires_grad]
        if len(trainable) > 0:
            params = trainable
    elif mode == "base_model":
        # Keep the full parameter set in declared model order.
        params = list(params)
    else:
        raise ValueError(f"Unknown shared parameter mode: {mode}")

    if len(params) == 0:
        params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        params = list(model.parameters())
    if len(params) == 0:
        raise ValueError("Model has no shared parameters.")
    return params
