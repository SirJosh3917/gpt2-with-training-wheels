from typing import Any
import torch.nn as nn


def load_from_state_dict(model: nn.Module, state_dict: Any, prefix: str):
    """
    It's more proper to have classes that inherit from `nn.Module`, and then
    call `model.load_from_state_dict()` to auto-propagate the saved data
    throughout the various layers.

    Because we're aiming to build a simple intuition, this bypasses those
    fancy helpers and directly inserts any data for a given layer into the
    exact layer that needs it.
    """

    local_metadata = []
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    model._load_from_state_dict(
        state_dict, prefix=prefix+".",
        strict=True,
        local_metadata=local_metadata,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        error_msgs=error_msgs
    )

    assert len(error_msgs) == 0, \
        f"There should be no error messages, but found some: {error_msgs}"
    assert len(missing_keys) == 0, \
        f"There should be no missing keys, but found some: {missing_keys}"
    assert len(unexpected_keys) == 0, \
        f"There should be no unexpected keys, but found some: {unexpected_keys}"
