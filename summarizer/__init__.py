"""Core building blocks for the abstractive summarization pipeline."""

from .callbacks import EvaluationInspectionCallback, IterationPauseCallback, SampleCountLoggingCallback
from .config import TrainingConfig, parse_training_args
from .datasets import (
    ColumnMapping,
    compute_lead_baseline,
    infer_columns,
    lead_n,
    preprocess_function,
)
from .inspection import (
    collect_samples_for_inspection,
    save_inspection_artifacts,
)
from .logging_utils import configure_logging, record_cli_invocation, tensorboard_writer_context
from .runtime import get_device, seed_everything
from .training import build_compute_metrics, build_training_arguments, create_trainer

__all__ = [
    "ColumnMapping",
    "TrainingConfig",
    "EvaluationInspectionCallback",
    "IterationPauseCallback",
    "SampleCountLoggingCallback",
    "build_compute_metrics",
    "build_training_arguments",
    "collect_samples_for_inspection",
    "compute_lead_baseline",
    "configure_logging",
    "create_trainer",
    "get_device",
    "infer_columns",
    "lead_n",
    "seed_everything",
    "parse_training_args",
    "preprocess_function",
    "save_inspection_artifacts",
    "record_cli_invocation",
    "tensorboard_writer_context",
]
