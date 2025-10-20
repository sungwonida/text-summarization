"""Train and evaluate a baseline abstractive summarization model for Week 3.

This script fine-tunes a pretrained seq2seq model (default: distilbart-cnn-12-6)
from Hugging Face Transformers on an English summarization dataset (default:
cnn_dailymail). It also reports a simple extractive "lead-3" baseline and saves a
summary of the evaluation metrics for portfolio documentation.

The script is deliberately configurable through command line arguments so that it
can scale from quick smoke-tests on a laptop GPU/Apple Silicon "mps" device to
longer training runs on a full GPU. It only depends on standard Hugging Face
libraries, PyTorch, and Evaluate.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import evaluate
import torch
from datasets import load_dataset
import matplotlib

matplotlib.use("Agg")

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import get_last_checkpoint

from week3.core import (
    EvaluationInspectionCallback,
    IterationPauseCallback,
    TrainingConfig,
    build_compute_metrics,
    build_training_arguments,
    collect_samples_for_inspection,
    compute_lead_baseline,
    configure_logging,
    create_trainer,
    get_device,
    infer_columns,
    parse_training_args,
    preprocess_function,
    save_inspection_artifacts,
    tensorboard_writer_context,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> TrainingConfig:
    return parse_training_args()


def serialize_inspection_sample(sample: Optional[Dict[str, object]]):
    if not sample:
        return None
    return {
        "sample_index": sample.get("sample_index"),
        "article": sample.get("article"),
        "reference_summary": sample.get("reference_summary"),
        "model_summary": sample.get("model_summary"),
        "lead_summary": sample.get("lead_summary"),
        "metrics": sample.get("metrics"),
        "primary_score": sample.get("primary_score"),
    }


def main():
    configure_logging()
    config = parse_args()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_argument = config.resume_from_checkpoint
    resume_path: Optional[str] = None
    try:
        last_checkpoint = get_last_checkpoint(str(output_dir))
    except Exception:  # pragma: no cover - defensive guard against HF internals
        last_checkpoint = None

    if resume_argument:
        if isinstance(resume_argument, str):
            resume_path = resume_argument
        else:
            if last_checkpoint:
                resume_path = last_checkpoint
            else:
                LOGGER.warning(
                    "--resume-from-checkpoint was provided but no checkpoint was found in %s; starting from scratch.",
                    output_dir,
                )
    elif last_checkpoint:
        LOGGER.warning(
            "Existing checkpoints detected in %s. Pass --resume-from-checkpoint to resume instead of overwriting.",
            output_dir,
        )

    if config.tensorboard_enabled():
        log_dir = Path(config.logging_dir) if config.logging_dir else output_dir / "runs"
        config.logging_dir = str(log_dir)
    elif config.logging_dir is not None:
        config.logging_dir = str(Path(config.logging_dir))

    device = get_device()
    torch.manual_seed(config.seed)

    LOGGER.info("Loading dataset %s (%s)", config.dataset_name, config.dataset_config)
    raw_datasets = load_dataset(config.dataset_name, config.dataset_config)

    columns = infer_columns(raw_datasets, config.text_column, config.summary_column)
    LOGGER.info("Using text column '%s' and summary column '%s'", columns.text_column, columns.summary_column)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    model.to(device)

    preprocess = preprocess_function(
        tokenizer,
        text_column=columns.text_column,
        summary_column=columns.summary_column,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
    )

    reference_split = raw_datasets.get("train") or next(iter(raw_datasets.values()))
    tokenized_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        remove_columns=reference_split.column_names,
        desc="Tokenizing",
    )

    if config.max_train_samples:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(config.max_train_samples))
    if config.max_eval_samples:
        for split_name in ("validation", "test"):
            if split_name in tokenized_datasets:
                tokenized_datasets[split_name] = tokenized_datasets[split_name].select(
                    range(min(config.max_eval_samples, len(tokenized_datasets[split_name])))
                )

    inspection_dataset = raw_datasets.get("test") or raw_datasets.get("validation")
    if inspection_dataset is not None and config.max_eval_samples:
        inspection_dataset = inspection_dataset.select(
            range(min(config.max_eval_samples, len(inspection_dataset)))
        )

    label_pad_token_id = -100 if tokenizer.pad_token_id is not None else 0
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=label_pad_token_id)

    rouge = evaluate.load("rouge")
    compute_metrics = build_compute_metrics(tokenizer, rouge)

    with tensorboard_writer_context(config, output_dir) as summary_writer:
        inspection_mode = config.sample_inspection_mode
        inspection_enabled = inspection_mode != "off"
        inspection_dir: Optional[Path] = None
        if inspection_enabled and inspection_mode == "tensorboard" and summary_writer is None:
            LOGGER.warning(
                "TensorBoard inspection requested but TensorBoard logging is disabled; falling back to debug path."
            )
            inspection_mode = "debug_path"
        if inspection_enabled:
            if inspection_mode == "debug_path":
                base_dir = (
                    Path(config.sample_inspection_dir)
                    if config.sample_inspection_dir
                    else output_dir / "debug_samples"
                )
                base_dir.mkdir(parents=True, exist_ok=True)
                inspection_dir = base_dir
            elif config.sample_inspection_dir:
                inspection_dir = Path(config.sample_inspection_dir)
                inspection_dir.mkdir(parents=True, exist_ok=True)

        training_args = build_training_arguments(config, output_dir)
        trainer = create_trainer(
            model=model,
            training_args=training_args,
            tokenized_datasets=tokenized_datasets,
            tokenizer=tokenizer,
            data_collator=data_collator,
            config=config,
            compute_metrics=compute_metrics,
        )

        if config.iteration_idle_time > 0 and config.iteration_idle_interval > 0:
            trainer.add_callback(
                IterationPauseCallback(
                    pause_seconds=config.iteration_idle_time,
                    interval=config.iteration_idle_interval,
                )
            )

        inspection_callback = None
        if inspection_enabled:
            inspection_callback = EvaluationInspectionCallback(
                config=config,
                columns=columns,
                dataset=inspection_dataset,
                model=model,
                tokenizer=tokenizer,
                device=device,
                rouge_metric=rouge,
                inspection_dir=inspection_dir,
                inspection_mode=inspection_mode,
                summary_writer=summary_writer,
            )
            trainer.add_callback(inspection_callback)

        if summary_writer is not None:
            summary_writer.add_text(
                "config/dataset",
                json.dumps(
                    {
                        "name": config.dataset_name,
                        "config": config.dataset_config,
                        "text_column": columns.text_column,
                        "summary_column": columns.summary_column,
                        "max_train_samples": config.max_train_samples,
                        "max_eval_samples": config.max_eval_samples,
                    },
                    indent=2,
                ),
                global_step=0,
            )

        if tokenized_datasets.get("train") is not None:
            LOGGER.info("Starting training")
            if resume_path:
                LOGGER.info("Resuming from checkpoint %s", resume_path)
                trainer.train(resume_from_checkpoint=resume_path)
            else:
                trainer.train()
        else:
            LOGGER.warning("No training split found; skipping training")

        LOGGER.info("Saving model and tokenizer to %s", output_dir)
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        eval_split = tokenized_datasets.get("validation") or tokenized_datasets.get("test")
        eval_metrics: Optional[Dict[str, float]] = None
        if eval_split is not None:
            LOGGER.info("Running evaluation")
            eval_metrics = trainer.evaluate(eval_dataset=eval_split, max_length=config.val_max_target_length)
            LOGGER.info("Evaluation metrics: %s", eval_metrics)
        else:
            LOGGER.warning("No evaluation split found; skipping evaluation")

        LOGGER.info("Computing lead-%d baseline", config.baseline_sentences)
        baseline_scores = None
        if eval_split is not None:
            original_eval_split = raw_datasets.get("validation") or raw_datasets.get("test")
            if config.max_eval_samples and original_eval_split is not None:
                original_eval_split = original_eval_split.select(
                    range(min(config.max_eval_samples, len(original_eval_split)))
                )
            if original_eval_split is not None:
                baseline_scores = compute_lead_baseline(
                    original_eval_split,
                    text_column=columns.text_column,
                    summary_column=columns.summary_column,
                    num_sentences=config.baseline_sentences,
                )
                LOGGER.info("Lead-%d baseline scores: %s", config.baseline_sentences, baseline_scores)

        qualitative_samples: List[Dict[str, object]] = []
        best_inspection_sample: Optional[Dict[str, object]] = None
        worst_inspection_sample: Optional[Dict[str, object]] = None
        bertscore_metric = None
        if inspection_enabled:
            if inspection_callback is not None:
                bertscore_metric = inspection_callback.get_bertscore_metric()
            if bertscore_metric is None:
                try:
                    bertscore_metric = evaluate.load("bertscore")
                except Exception as exc:  # pragma: no cover - network/model download issues
                    LOGGER.warning("Unable to load BERTScore metric for sample inspection: %s", exc)
                    bertscore_metric = None

        if eval_split is not None:
            LOGGER.info("Generating qualitative samples")
            qualitative_samples, best_inspection_sample, worst_inspection_sample = collect_samples_for_inspection(
                inspection_dataset,
                columns=columns,
                model=model,
                tokenizer=tokenizer,
                device=device,
                config=config,
                rouge_metric=rouge,
                bertscore_metric=bertscore_metric,
                inspection_enabled=inspection_enabled,
            )

        global_step = getattr(trainer.state, "global_step", 0)
        if summary_writer is not None:
            if eval_metrics:
                for key, value in eval_metrics.items():
                    if isinstance(value, (int, float)):
                        summary_writer.add_scalar(f"eval/{key}", value, global_step=global_step)
            if baseline_scores:
                for key, value in baseline_scores.items():
                    if isinstance(value, (int, float)):
                        summary_writer.add_scalar(f"baseline/{key}", value, global_step=global_step)
            for idx, sample in enumerate(qualitative_samples):
                summary_writer.add_text(
                    f"samples/{idx}",
                    json.dumps(sample, indent=2),
                    global_step=global_step,
                )

        if inspection_enabled:
            save_inspection_artifacts(
                best_inspection_sample,
                "good",
                inspection_dir,
                global_step,
                summary_writer,
                inspection_mode,
            )
            save_inspection_artifacts(
                worst_inspection_sample,
                "bad",
                inspection_dir,
                global_step,
                summary_writer,
                inspection_mode,
            )

        inspection_summary = None
        if inspection_enabled:
            inspection_summary = {
                "mode": inspection_mode,
                "good": serialize_inspection_sample(best_inspection_sample),
                "bad": serialize_inspection_sample(worst_inspection_sample),
            }

        report = {
            "dataset": {
                "name": config.dataset_name,
                "config": config.dataset_config,
                "text_column": columns.text_column,
                "summary_column": columns.summary_column,
                "max_train_samples": config.max_train_samples,
                "max_eval_samples": config.max_eval_samples,
            },
            "model": config.model_name,
            "training_args": asdict(config),
            "evaluation": eval_metrics,
            "baseline": baseline_scores,
            "samples": qualitative_samples,
        }

        if inspection_summary:
            report["inspection"] = inspection_summary

        report_path = output_dir / "evaluation_report.json"
        LOGGER.info("Writing evaluation report to %s", report_path)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
