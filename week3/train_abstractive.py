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

import argparse
import inspect
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
import evaluate
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        default="cnn_dailymail",
        help="Hugging Face dataset identifier to use for fine-tuning.",
    )
    parser.add_argument(
        "--dataset-config",
        default="3.0.0",
        help="Dataset configuration name (for datasets with multiple configurations).",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help=(
            "Name of the source text column. If omitted the script will try to "
            "infer a standard column (article or document)."
        ),
    )
    parser.add_argument(
        "--summary-column",
        default=None,
        help=(
            "Name of the summary/target column. If omitted the script will try to "
            "infer a standard column (highlights or summary)."
        ),
    )
    parser.add_argument(
        "--model-name",
        default="sshleifer/distilbart-cnn-12-6",
        help="Pretrained model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/week3",
        help="Directory where checkpoints and evaluation artifacts are stored.",
    )
    parser.add_argument(
        "--max-source-length",
        type=int,
        default=512,
        help="Maximum number of tokens for the encoder input.",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=128,
        help="Maximum number of tokens for the decoder target.",
    )
    parser.add_argument(
        "--val-max-target-length",
        type=int,
        default=None,
        help=(
            "Maximum number of tokens used for validation generation. "
            "Defaults to --max-target-length."
        ),
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=2,
        help="Training batch size per device (GPU/MPS/CPU).",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=4,
        help="Evaluation batch size per device (GPU/MPS/CPU).",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Number of epochs to train. Fractional values enable quick smoke tests.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=25,
        help="Number of steps between logging updates.",
    )
    parser.add_argument(
        "--evaluation-strategy",
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="Evaluation frequency.",
    )
    parser.add_argument(
        "--save-strategy",
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="Checkpointing frequency.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before performing a backward pass.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="If set, truncates the training set to the specified number of samples.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=500,
        help=(
            "If set, truncates the validation/test sets to this number of samples. "
            "Defaults to 500 to keep evaluation lightweight."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--predict-with-generate",
        action="store_true",
        help="Use generation for evaluation metrics (recommended for summarization).",
    )
    parser.add_argument(
        "--generation-num-beams",
        type=int,
        default=4,
        help="Number of beams for beam search during evaluation/prediction.",
    )
    parser.add_argument(
        "--baseline-sentences",
        type=int,
        default=3,
        help="Number of leading sentences used for the extractive baseline.",
    )
    parser.add_argument(
        "--num-samples-for-report",
        type=int,
        default=3,
        help="Number of qualitative samples to store in the evaluation report.",
    )

    args = parser.parse_args()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length
    return args


def configure_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        LOGGER.info("Using MPS device")
        return torch.device("mps")
    if torch.cuda.is_available():
        LOGGER.info("Using CUDA device")
        return torch.device("cuda")
    LOGGER.info("Using CPU device")
    return torch.device("cpu")


@dataclass
class ColumnMapping:
    text_column: str
    summary_column: str


def infer_columns(dataset: DatasetDict, text_column: str | None, summary_column: str | None) -> ColumnMapping:
    if text_column and summary_column:
        return ColumnMapping(text_column=text_column, summary_column=summary_column)

    sample_split = dataset.get("train") or next(iter(dataset.values()))
    candidate_mappings = [
        ("article", "highlights"),
        ("document", "summary"),
        ("text", "summary"),
    ]
    for candidate_text, candidate_summary in candidate_mappings:
        if candidate_text in sample_split.column_names and candidate_summary in sample_split.column_names:
            text_column = text_column or candidate_text
            summary_column = summary_column or candidate_summary
            break

    if text_column is None or summary_column is None:
        raise ValueError(
            "Could not infer text/summary columns. Please specify --text-column and --summary-column explicitly."
        )

    return ColumnMapping(text_column=text_column, summary_column=summary_column)


SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


def lead_n(text: str, n: int) -> str:
    """Return the concatenation of the first ``n`` sentences from ``text``."""
    if not text:
        return ""
    sentences = SENTENCE_SPLIT_REGEX.split(text.strip())
    return " ".join(sentences[:n]).strip()


def compute_lead_baseline(
    dataset_split,
    text_column: str,
    summary_column: str,
    num_sentences: int,
) -> Dict[str, float]:
    metric = evaluate.load("rouge")
    predictions = []
    references = []

    for record in dataset_split:
        predictions.append(lead_n(record[text_column], num_sentences))
        references.append(record[summary_column])

    scores = metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    return scores


def preprocess_function(tokenizer, text_column: str, summary_column: str, max_source_length: int, max_target_length: int):
    def _preprocess(batch):
        inputs = batch[text_column]
        targets = batch[summary_column]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _preprocess


def postprocess_text(preds: Iterable[str], labels: Iterable[str]) -> Tuple[List[str], List[str]]:
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels


def build_training_arguments(args: argparse.Namespace, output_dir: Path) -> Seq2SeqTrainingArguments:
    """Create ``Seq2SeqTrainingArguments`` while remaining compatible with multiple versions."""

    base_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "logging_steps": args.logging_steps,
        "num_train_epochs": args.num_train_epochs,
        "predict_with_generate": args.predict_with_generate,
        "generation_max_length": args.val_max_target_length,
        "generation_num_beams": args.generation_num_beams,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "report_to": ["none"],
        "seed": args.seed,
    }

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    valid_params = set(signature.parameters.keys())

    filtered_kwargs: Dict[str, object] = {}
    skipped_keys: List[str] = []

    for key, value in base_kwargs.items():
        if key in valid_params:
            filtered_kwargs[key] = value
        else:
            skipped_keys.append(key)

    for skipped in skipped_keys:
        LOGGER.warning("Dropping unsupported Seq2SeqTrainingArguments option '%s'", skipped)

    eval_value = args.evaluation_strategy
    if "evaluation_strategy" in valid_params:
        filtered_kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in valid_params:
        filtered_kwargs["eval_strategy"] = eval_value
    elif "evaluate_during_training" in valid_params:
        filtered_kwargs["evaluate_during_training"] = eval_value != "no"
        if eval_value == "steps" and "eval_steps" in valid_params:
            filtered_kwargs["eval_steps"] = max(1, args.logging_steps)
    elif eval_value != "no":
        LOGGER.warning(
            "Seq2SeqTrainingArguments version does not support evaluation strategy; "
            "evaluation during training will be disabled."
        )

    save_value = args.save_strategy
    if "save_strategy" in valid_params:
        filtered_kwargs["save_strategy"] = save_value
    elif save_value != "no" and "save_steps" in valid_params:
        filtered_kwargs["save_steps"] = max(1, args.logging_steps)
    elif save_value != "no":
        LOGGER.warning(
            "Seq2SeqTrainingArguments version does not support save strategy; "
            "default checkpointing behavior will be used."
        )

    if "report_to" not in valid_params and "report_to" in filtered_kwargs:
        filtered_kwargs.pop("report_to", None)

    return Seq2SeqTrainingArguments(**filtered_kwargs)


def main():
    configure_logging()
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    torch.manual_seed(args.seed)

    LOGGER.info("Loading dataset %s (%s)", args.dataset_name, args.dataset_config)
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config)

    columns = infer_columns(raw_datasets, args.text_column, args.summary_column)
    LOGGER.info("Using text column '%s' and summary column '%s'", columns.text_column, columns.summary_column)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(device)

    preprocess = preprocess_function(
        tokenizer,
        text_column=columns.text_column,
        summary_column=columns.summary_column,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    reference_split = raw_datasets.get("train") or next(iter(raw_datasets.values()))
    tokenized_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        remove_columns=reference_split.column_names,
        desc="Tokenizing",
    )

    if args.max_train_samples:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(args.max_train_samples))
    if args.max_eval_samples:
        for split_name in ("validation", "test"):
            if split_name in tokenized_datasets:
                tokenized_datasets[split_name] = tokenized_datasets[split_name].select(
                    range(min(args.max_eval_samples, len(tokenized_datasets[split_name])))
                )

    label_pad_token_id = -100 if tokenizer.pad_token_id is not None else 0
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=label_pad_token_id)

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = [[label for label in label_row if label != -100] for label_row in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = sum(prediction_lens) / len(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}

    training_args = build_training_arguments(args, output_dir)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=tokenized_datasets.get("validation") or tokenized_datasets.get("test"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.predict_with_generate else None,
    )

    if tokenized_datasets.get("train") is not None:
        LOGGER.info("Starting training")
        trainer.train()
    else:
        LOGGER.warning("No training split found; skipping training")

    LOGGER.info("Saving model and tokenizer to %s", output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    eval_split = tokenized_datasets.get("validation") or tokenized_datasets.get("test")
    eval_metrics: Dict[str, float] | None = None
    if eval_split is not None:
        LOGGER.info("Running evaluation")
        eval_metrics = trainer.evaluate(eval_dataset=eval_split, max_length=args.val_max_target_length)
        LOGGER.info("Evaluation metrics: %s", eval_metrics)
    else:
        LOGGER.warning("No evaluation split found; skipping evaluation")

    LOGGER.info("Computing lead-%d baseline", args.baseline_sentences)
    baseline_scores = None
    if eval_split is not None:
        original_eval_split = raw_datasets.get("validation") or raw_datasets.get("test")
        if args.max_eval_samples:
            original_eval_split = original_eval_split.select(
                range(min(args.max_eval_samples, len(original_eval_split)))
            )
        baseline_scores = compute_lead_baseline(
            original_eval_split,
            text_column=columns.text_column,
            summary_column=columns.summary_column,
            num_sentences=args.baseline_sentences,
        )
        LOGGER.info("Lead-%d baseline scores: %s", args.baseline_sentences, baseline_scores)

    qualitative_samples: List[Dict[str, str]] = []
    if eval_split is not None:
        LOGGER.info("Generating qualitative samples")
        eval_dataset_for_samples = raw_datasets.get("test") or raw_datasets.get("validation")
        if eval_dataset_for_samples is not None:
            if args.max_eval_samples:
                eval_dataset_for_samples = eval_dataset_for_samples.select(
                    range(min(args.max_eval_samples, len(eval_dataset_for_samples)))
                )
            sample_indices = list(range(min(args.num_samples_for_report, len(eval_dataset_for_samples))))
            for idx in sample_indices:
                record = eval_dataset_for_samples[idx]
                input_text = record[columns.text_column]
                reference_summary = record[columns.summary_column]
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=args.max_source_length,
                    truncation=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_length=args.val_max_target_length,
                        num_beams=args.generation_num_beams,
                    )
                predicted_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                qualitative_samples.append(
                    {
                        "article": input_text,
                        "reference_summary": reference_summary,
                        "model_summary": predicted_summary,
                    }
                )

    report = {
        "dataset": {
            "name": args.dataset_name,
            "config": args.dataset_config,
            "text_column": columns.text_column,
            "summary_column": columns.summary_column,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
        },
        "model": args.model_name,
        "training_args": vars(args),
        "evaluation": eval_metrics,
        "baseline": baseline_scores,
        "samples": qualitative_samples,
    }

    report_path = output_dir / "evaluation_report.json"
    LOGGER.info("Writing evaluation report to %s", report_path)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
