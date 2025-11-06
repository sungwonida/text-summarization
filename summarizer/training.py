"""Training-related helpers for assembling the Hugging Face Trainer."""

from __future__ import annotations

import inspect
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Sampler
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import ExportableState
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, SaveStrategy

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
        self._samples_seen_initialized = False
        self._checkpoint_registry: Dict[int, str] = {}
        super().__init__(*args, **kwargs)
        self._bootstrap_checkpoint_registry()

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

    def _initialize_samples_seen(self) -> None:
        if self._samples_seen_initialized:
            return
        existing = getattr(self.state, "samples_seen", None)
        if existing is None:
            prior_steps = int(getattr(self.state, "global_step", 0) or 0)
            train_batch_size = getattr(self.state, "train_batch_size", None)
            if train_batch_size is None:
                train_batch_size = getattr(self.args, "train_batch_size", None)
            if train_batch_size is None:
                per_device = getattr(self.args, "per_device_train_batch_size", 1)
                world_size = max(1, getattr(self.args, "world_size", 1))
                train_batch_size = per_device * world_size
            grad_accum = max(1, getattr(self.args, "gradient_accumulation_steps", 1))
            existing = int(prior_steps * train_batch_size * grad_accum)
        if existing is None:
            existing = 0
        self.state.samples_seen = int(existing)
        self._samples_seen_initialized = True

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._initialize_samples_seen()
        observed_batch = find_batch_size(inputs)
        total_samples = None
        if observed_batch is not None:
            world_size = max(1, getattr(self.args, "world_size", 1))
            total_samples = int(observed_batch) * world_size
        else:
            fallback_batch = getattr(self.args, "train_batch_size", None)
            if fallback_batch is None:
                per_device = getattr(self.args, "per_device_train_batch_size", None)
                if per_device is not None:
                    world_size = max(1, getattr(self.args, "world_size", 1))
                    fallback_batch = int(per_device) * world_size
            if fallback_batch is not None:
                total_samples = int(fallback_batch)

        if total_samples is not None:
            previous = int(getattr(self.state, "samples_seen", 0) or 0)
            self.state.samples_seen = previous + total_samples
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    # Hugging Face Trainer overrides -------------------------------------------------
    def _bootstrap_checkpoint_registry(self) -> None:
        run_dir = Path(getattr(self.args, "output_dir", ""))
        if not run_dir.is_dir():
            return

        for checkpoint_dir in sorted(run_dir.glob(f"{PREFIX_CHECKPOINT_DIR}-*")):
            state_path = checkpoint_dir / TRAINER_STATE_NAME
            if not state_path.is_file():
                continue
            try:
                with state_path.open("r", encoding="utf-8") as handle:
                    state_payload = json.load(handle)
            except Exception:
                continue
            step = state_payload.get("global_step")
            if isinstance(step, int):
                self._checkpoint_registry[int(step)] = str(checkpoint_dir)

        best_step = getattr(self.state, "best_global_step", None)
        if isinstance(best_step, int) and best_step in self._checkpoint_registry:
            self.state.best_model_checkpoint = self._checkpoint_registry[best_step]

    def _current_samples_seen(self) -> int:
        samples_seen = getattr(self.state, "samples_seen", None)
        if samples_seen is not None:
            return int(samples_seen)

        train_batch_size = getattr(self.state, "train_batch_size", None)
        if train_batch_size is None:
            train_batch_size = getattr(self.args, "train_batch_size", None)
        if train_batch_size is None:
            per_device = getattr(self.args, "per_device_train_batch_size", 1)
            world_size = max(1, getattr(self.args, "world_size", 1))
            train_batch_size = per_device * world_size

        grad_accum = max(1, getattr(self.args, "gradient_accumulation_steps", 1))
        global_step = int(getattr(self.state, "global_step", 0) or 0)
        return int(global_step * grad_accum * train_batch_size)

    def _save_checkpoint(self, model, trial):
        checkpoint_suffix = self._current_samples_seen()
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{checkpoint_suffix}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        current_step = int(getattr(self.state, "global_step", 0) or 0)
        self._checkpoint_registry[current_step] = output_dir

        best_step = getattr(self.state, "best_global_step", None)
        if isinstance(best_step, int):
            if best_step == current_step:
                self.state.best_model_checkpoint = output_dir
            elif best_step in self._checkpoint_registry:
                self.state.best_model_checkpoint = self._checkpoint_registry[best_step]

        if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
            best_checkpoint_dir = self.state.best_model_checkpoint
            if best_checkpoint_dir and os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if not self.args.save_only_model:
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            self._save_rng_state(output_dir)

        if self.args.should_save:
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


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
