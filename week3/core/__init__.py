"""Core building blocks for the week 3 abstractive summarization pipeline."""

from .callbacks import EvaluationInspectionCallback, IterationPauseCallback
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
from .logging_utils import configure_logging, tensorboard_writer_context
from .runtime import get_device
from .training import build_compute_metrics, build_training_arguments, create_trainer

__all__ = [
    "ColumnMapping",
    "TrainingConfig",
    "EvaluationInspectionCallback",
    "IterationPauseCallback",
    "build_compute_metrics",
    "build_training_arguments",
    "collect_samples_for_inspection",
    "compute_lead_baseline",
    "configure_logging",
    "create_trainer",
    "get_device",
    "infer_columns",
    "lead_n",
    "parse_training_args",
    "preprocess_function",
    "save_inspection_artifacts",
    "tensorboard_writer_context",
]

