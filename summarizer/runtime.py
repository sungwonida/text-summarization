"""Runtime helpers such as device selection."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Configure all supported libraries for deterministic behaviour."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        # Older PyTorch builds or unsupported ops fall back gracefully.
        LOGGER.debug("Deterministic algorithms unavailable for this configuration.")

    try:
        from transformers import set_seed as hf_set_seed  # Lazy import to avoid circular deps.

        hf_set_seed(seed)
    except Exception:  # pragma: no cover - transformers always available in runtime, but fail safe.
        LOGGER.debug("Unable to propagate seed to Transformers helpers.")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        LOGGER.info("Using MPS device")
        return torch.device("mps")
    if torch.cuda.is_available():
        LOGGER.info("Using CUDA device")
        return torch.device("cuda")
    LOGGER.info("Using CPU device")
    return torch.device("cpu")
