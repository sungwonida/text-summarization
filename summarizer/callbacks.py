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
from .inspection import collect_samples_for_inspection, save_inspection_artifacts

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

    def _ensure_bertscore_metric(self):
        if not self.inspection_enabled or self.bertscore_metric is not None:
            return
        try:  # pragma: no cover - optional dependency download path
            self.bertscore_metric = evaluate.load("bertscore")
        except Exception as exc:
            LOGGER.warning("Unable to load BERTScore metric for sample inspection: %s", exc)
            self.bertscore_metric = None

    def get_bertscore_metric(self):
        self._ensure_bertscore_metric()
        return self.bertscore_metric

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

        return control

