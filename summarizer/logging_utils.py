"""Logging helpers for the training pipeline."""

from __future__ import annotations

import logging
import shlex
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Sequence

from .config import TrainingConfig, should_enable_tensorboard

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def record_cli_invocation(output_dir: Path, argv: Optional[Sequence[str]] = None, *, filename: str = "cli_invocations.txt") -> Path:
    """Append the current command-line invocation to a reproducibility log."""

    args = list(argv) if argv is not None else list(sys.argv)
    command = " ".join(shlex.quote(entry) for entry in [sys.executable, *args] if entry)
    timestamp = datetime.now(timezone.utc).isoformat()

    log_path = output_dir / filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {command}\n")

    return log_path


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
