"""Small neural-network helpers required by the reach actor."""

from __future__ import annotations

import torch


def resolve_nn_activation(name: str) -> torch.nn.Module:
    activations = {
        "elu": torch.nn.ELU,
        "selu": torch.nn.SELU,
        "relu": torch.nn.ReLU,
        "crelu": torch.nn.CELU,
        "lrelu": torch.nn.LeakyReLU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "identity": torch.nn.Identity,
    }
    try:
        return activations[name]()
    except KeyError as error:
        raise ValueError(f"Invalid activation function {name!r}") from error
