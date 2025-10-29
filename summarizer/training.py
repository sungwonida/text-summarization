"""Training-related helpers for assembling the Hugging Face Trainer."""

from __future__ import annotations

import inspect
import logging
import random
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Sampler
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from .config import TrainingConfig
from .inspection import postprocess_text

LOGGER = logging.getLogger(__name__)


class RandomSubsetSampler(Sampler[int]):
    """Sampler that draws a new random subset of indices on every epoch."""

    def __init__(self, data_source, *, num_samples: int, seed: int) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        self.data_source = data_source
        self.num_samples = num_samples
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        dataset_size = len(self.data_source)
        requested = min(self.num_samples, dataset_size)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        self._epoch += 1

        # torch.randperm provides a reproducible permutation based on the generator seed.
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        return iter(indices[:requested])

    def __len__(self) -> int:
        return min(self.num_samples, len(self.data_source))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch


class CappedSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer variant that caps per-epoch training samples via random subsets."""

    def __init__(self, *args, max_train_samples_per_epoch: Optional[int] = None, **kwargs):
        self._max_train_samples_per_epoch = max_train_samples_per_epoch
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, *args, **kwargs):
        sampler = super()._get_train_sampler(*args, **kwargs)
        if self._max_train_samples_per_epoch is None:
            return sampler
        if sampler is None:
            return None
        dataset = args[0] if args else getattr(self, "train_dataset", None)
        if dataset is None:
            return sampler
        dataset_size = len(dataset)
        if dataset_size <= self._max_train_samples_per_epoch:
            return sampler
        if getattr(self.args, "world_size", 1) > 1:
            LOGGER.warning(
                "Capped per-epoch sampling is not currently supported with distributed training; "
                "falling back to the default sampler."
            )
            return sampler

        return RandomSubsetSampler(
            dataset,
            num_samples=self._max_train_samples_per_epoch,
            seed=getattr(self.args, "seed", 0),
        )


def build_training_arguments(config: TrainingConfig, output_dir: Path) -> Seq2SeqTrainingArguments:
    """Create ``Seq2SeqTrainingArguments`` while remaining compatible with multiple versions."""

    base_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": config.resume_from_checkpoint is None,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "weight_decay": config.weight_decay,
        "warmup_steps": config.warmup_steps,
        "logging_steps": config.logging_steps,
        "num_train_epochs": config.num_train_epochs,
        "predict_with_generate": config.predict_with_generate,
        "generation_max_length": config.val_max_target_length,
        "generation_num_beams": config.generation_num_beams,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "report_to": config.report_to,
        "seed": config.seed,
    }

    if config.logging_dir is not None:
        base_kwargs["logging_dir"] = config.logging_dir

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    valid_params = set(signature.parameters.keys())

    filtered_kwargs: Dict[str, object] = {}
    skipped_keys = []

    for key, value in base_kwargs.items():
        if key in valid_params:
            filtered_kwargs[key] = value
        else:
            skipped_keys.append(key)

    for skipped in skipped_keys:
        LOGGER.warning("Dropping unsupported Seq2SeqTrainingArguments option '%s'", skipped)

    eval_value = config.evaluation_strategy
    if "evaluation_strategy" in valid_params:
        filtered_kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in valid_params:
        filtered_kwargs["eval_strategy"] = eval_value
    elif "evaluate_during_training" in valid_params:
        filtered_kwargs["evaluate_during_training"] = eval_value != "no"
        if eval_value == "steps" and "eval_steps" in valid_params:
            filtered_kwargs["eval_steps"] = max(1, config.logging_steps)
    elif eval_value != "no":
        LOGGER.warning(
            "Seq2SeqTrainingArguments version does not support evaluation strategy; "
            "evaluation during training will be disabled."
        )

    save_value = config.save_strategy
    if "save_strategy" in valid_params:
        filtered_kwargs["save_strategy"] = save_value
    elif save_value != "no" and "save_steps" in valid_params:
        filtered_kwargs["save_steps"] = max(1, config.logging_steps)
    elif save_value != "no":
        LOGGER.warning(
            "Seq2SeqTrainingArguments version does not support save strategy; "
            "default checkpointing behavior will be used."
        )

    if "report_to" not in valid_params and "report_to" in filtered_kwargs:
        filtered_kwargs.pop("report_to", None)

    return Seq2SeqTrainingArguments(**filtered_kwargs)


def build_compute_metrics(tokenizer, rouge_metric, *, seed: int | None = None):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            if hasattr(rouge_metric, "seed"):
                rouge_metric.seed = seed

        labels = [[label for label in label_row if label != -100] for label_row in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = sum(prediction_lens) / len(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics


def create_trainer(
    *,
    model,
    training_args: Seq2SeqTrainingArguments,
    tokenized_datasets: Mapping[str, object],
    tokenizer,
    data_collator,
    config: TrainingConfig,
    compute_metrics=None,
):
    eval_dataset = tokenized_datasets.get("validation") or tokenized_datasets.get("test")
    return CappedSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if config.predict_with_generate else None,
        max_train_samples_per_epoch=config.max_train_samples,
    )
