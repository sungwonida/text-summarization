"""Trainer callbacks used by the training pipeline."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import evaluate
import torch
from transformers import TrainerCallback

from .config import TrainingConfig
from .datasets import ColumnMapping
from .inspection import collect_samples_for_inspection, inspect_dataset_sample, save_inspection_artifacts

LOGGER = logging.getLogger(__name__)


class IterationPauseCallback(TrainerCallback):
    """Pause the training/evaluation loop after a configurable number of steps."""

    def __init__(self, pause_seconds: float, interval: int) -> None:
        self.pause_seconds = pause_seconds
        self.interval = interval
        self._train_step_counter = 0
        self._prediction_step_counter = 0

    def _should_pause(self, counter: int) -> bool:
        return self.interval > 0 and self.pause_seconds > 0 and counter % self.interval == 0

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: D401 - HF callback signature
        self._train_step_counter = 0
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: D401 - HF callback signature
        if self.pause_seconds <= 0 or self.interval <= 0:
            return control
        self._train_step_counter += 1
        if self._should_pause(self._train_step_counter):
            time.sleep(self.pause_seconds)
        return control

    def on_prediction_step(self, args, state, control, **kwargs):  # noqa: D401 - HF callback signature
        if self.pause_seconds > 0 and self.interval > 0:
            self._prediction_step_counter += 1
            if self._should_pause(self._prediction_step_counter):
                time.sleep(self.pause_seconds)
        return control

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: D401 - HF callback signature
        self._prediction_step_counter = 0
        return control

    def on_predict(self, args, state, control, metrics=None, **kwargs):  # noqa: D401 - HF callback signature
        self._prediction_step_counter = 0
        return control


class EvaluationInspectionCallback(TrainerCallback):
    """Persist inspection samples after every evaluation during training."""

    def __init__(
        self,
        *,
        config: TrainingConfig,
        columns: ColumnMapping,
        dataset,
        model,
        tokenizer,
        device: torch.device,
        rouge_metric,
        inspection_dir: Optional[Path],
        inspection_mode: str,
        summary_writer,
    ) -> None:
        self.config = config
        self.columns = columns
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.rouge_metric = rouge_metric
        self.inspection_dir = inspection_dir
        self.inspection_mode = inspection_mode
        self.summary_writer = summary_writer
        self.inspection_enabled = inspection_mode != "off"
        self.bertscore_metric = None
        self._tracked_sample_collection_failed = False
        self.tracked_sample_index = self._resolve_tracked_sample_index()
        self.seed = config.seed
        if hasattr(self.rouge_metric, "seed"):
            self.rouge_metric.seed = self.seed
    def _ensure_bertscore_metric(self):
        if not self.inspection_enabled or self.bertscore_metric is not None:
            return
        try:  # pragma: no cover - optional dependency download path
            self.bertscore_metric = evaluate.load("bertscore")
            if hasattr(self.bertscore_metric, "seed"):
                self.bertscore_metric.seed = self.seed
        except Exception as exc:
            LOGGER.warning("Unable to load BERTScore metric for sample inspection: %s", exc)
            self.bertscore_metric = None

    def get_bertscore_metric(self):
        self._ensure_bertscore_metric()
        return self.bertscore_metric

    def _resolve_tracked_sample_index(self) -> Optional[int]:
        if self.dataset is None:
            if (
                self.config.tracked_sample_index is not None
                or self.config.tracked_sample_id_column is not None
            ):
                LOGGER.warning(
                    "Tracked sample requested but inspection dataset is unavailable; disabling tracked sample snapshots."
                )
            return None

        dataset_size = len(self.dataset)
        requested_index = self.config.tracked_sample_index
        if requested_index is not None:
            if requested_index >= dataset_size:
                LOGGER.warning(
                    "Tracked sample index %d is out of bounds for evaluation dataset of size %d; disabling tracked sample snapshots.",
                    requested_index,
                    dataset_size,
                )
                return None
            LOGGER.info("Tracking evaluation sample at index %d.", requested_index)
            return requested_index

        column = self.config.tracked_sample_id_column
        value = self.config.tracked_sample_id_value
        if column is None or value is None:
            return None

        if column not in self.dataset.column_names:
            LOGGER.warning(
                "Tracked sample column '%s' not present in evaluation dataset columns %s; disabling tracked sample snapshots.",
                column,
                list(self.dataset.column_names),
            )
            return None

        column_values = list(self.dataset[column])
        matches = [idx for idx, entry in enumerate(column_values) if entry == value]
        if not matches:
            LOGGER.warning(
                "Tracked sample value '%s' not found in column '%s'; disabling tracked sample snapshots.",
                value,
                column,
            )
            return None
        if len(matches) > 1:
            LOGGER.warning(
                "Multiple evaluation samples matched %s=%s; using the first match at index %d.",
                column,
                value,
                matches[0],
            )
        LOGGER.info("Tracking evaluation sample where %s=%s (index %d).", column, value, matches[0])
        return matches[0]

    def on_evaluate(self, args, state, control, **kwargs):  # noqa: D401 - HF callback signature
        if not self.inspection_enabled or self.dataset is None:
            return control

        self._ensure_bertscore_metric()

        qualitative_samples, best_sample, worst_sample = collect_samples_for_inspection(
            self.dataset,
            columns=self.columns,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            config=self.config,
            rouge_metric=self.rouge_metric,
            bertscore_metric=self.bertscore_metric,
            inspection_enabled=self.inspection_enabled,
            seed=self.seed,
        )

        global_step = getattr(state, "global_step", 0)
        save_inspection_artifacts(
            best_sample,
            "good",
            self.inspection_dir,
            global_step,
            self.summary_writer,
            self.inspection_mode,
        )
        save_inspection_artifacts(
            worst_sample,
            "bad",
            self.inspection_dir,
            global_step,
            self.summary_writer,
            self.inspection_mode,
        )

        if self.summary_writer is not None:
            for idx, sample in enumerate(qualitative_samples):
                self.summary_writer.add_text(
                    f"eval_samples/{idx}",
                    json.dumps(sample, indent=2),
                    global_step=global_step,
                )

        tracked_sample = None
        if self.tracked_sample_index is not None:
            _, tracked_sample = inspect_dataset_sample(
                self.dataset,
                sample_index=self.tracked_sample_index,
                columns=self.columns,
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                config=self.config,
                rouge_metric=self.rouge_metric,
                bertscore_metric=self.bertscore_metric,
                inspection_enabled=self.inspection_enabled,
                seed=self.seed,
            )
            if tracked_sample is None:
                if not self._tracked_sample_collection_failed:
                    LOGGER.warning(
                        "Unable to collect tracked sample at index %d during evaluation; snapshots will be skipped.",
                        self.tracked_sample_index,
                    )
                    self._tracked_sample_collection_failed = True
            else:
                if self.config.tracked_sample_id_column and self.config.tracked_sample_id_value:
                    tracked_sample["sample_identifier"] = (
                        f"{self.config.tracked_sample_id_column}={self.config.tracked_sample_id_value}"
                    )
                save_inspection_artifacts(
                    tracked_sample,
                    "tracked",
                    self.inspection_dir,
                    global_step,
                    self.summary_writer,
                    self.inspection_mode,
                )
        return control
