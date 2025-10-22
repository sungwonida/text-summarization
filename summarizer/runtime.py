"""Runtime helpers such as device selection."""

from __future__ import annotations

import logging

import torch

LOGGER = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        LOGGER.info("Using MPS device")
        return torch.device("mps")
    if torch.cuda.is_available():
        LOGGER.info("Using CUDA device")
        return torch.device("cuda")
    LOGGER.info("Using CPU device")
    return torch.device("cpu")

