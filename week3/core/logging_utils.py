"""Logging helpers for the training pipeline."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .config import TrainingConfig, should_enable_tensorboard

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


@contextmanager
def tensorboard_writer_context(config: TrainingConfig, output_dir: Path) -> Iterator[Optional[object]]:
    if not should_enable_tensorboard(config.report_to):
        yield None
        return

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "TensorBoard logging requested but tensorboard is not installed. Install it or pass --report-to none."
        ) from exc

    log_dir = Path(config.logging_dir) if config.logging_dir else output_dir / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    config.logging_dir = str(log_dir)
    LOGGER.info("TensorBoard logs will be written to %s", log_dir)

    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        yield writer
    finally:
        writer.flush()
        writer.close()

