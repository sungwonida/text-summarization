"""Command-line configuration handling for the training pipeline."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence


def parse_duration_to_seconds(raw_value: str) -> float:
    """Convert CLI duration strings like ``"0.5s"`` or ``"250ms"`` to seconds."""

    if isinstance(raw_value, (int, float)):
        return float(raw_value)

    value = str(raw_value).strip().lower()
    if value == "0":
        return 0.0

    match = re.fullmatch(r"(?P<number>\d+(?:\.\d+)?)(?P<unit>ms|s)?", value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid duration '{raw_value}'. Expected values like '0.5s' or '250ms'."
        )

    number = float(match.group("number"))
    unit = match.group("unit") or "s"

    if number == 0:
        return 0.0

    if unit == "ms":
        return number / 1000.0
    if unit == "s":
        return number

    raise argparse.ArgumentTypeError(
        f"Unsupported duration unit '{unit}' in value '{raw_value}'."
    )


def normalize_report_to(report_to: Iterable[str]) -> List[str]:
    normalized = [entry.lower() for entry in report_to if entry]
    if "none" in normalized and len(normalized) > 1:
        normalized = [entry for entry in normalized if entry != "none"]
    if not normalized:
        return ["none"]
    return normalized


def should_enable_tensorboard(report_to: Iterable[str]) -> bool:
    return any(entry == "tensorboard" for entry in report_to)


@dataclass
class TrainingConfig:
    dataset_name: str
    dataset_config: str
    text_column: Optional[str]
    summary_column: Optional[str]
    model_name: str
    output_dir: str
    resume_from_checkpoint: Optional[str | bool]
    max_source_length: int
    max_target_length: int
    val_max_target_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    num_train_epochs: float
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    logging_steps: int
    logging_dir: Optional[str]
    iteration_idle_time: float
    iteration_idle_interval: int
    evaluation_strategy: str
    save_strategy: str
    gradient_accumulation_steps: int
    report_to: List[str] = field(default_factory=list)
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    seed: int = 42
    predict_with_generate: bool = False
    generation_num_beams: int = 4
    baseline_sentences: int = 3
    num_samples_for_report: int = 3
    sample_inspection_mode: str = "debug_path"
    sample_inspection_dir: Optional[str] = None
    tracked_sample_index: Optional[int] = None
    tracked_sample_id_column: Optional[str] = None
    tracked_sample_id_value: Optional[str] = None

    def tensorboard_enabled(self) -> bool:
        return should_enable_tensorboard(self.report_to)


def build_arg_parser() -> argparse.ArgumentParser:
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
        default="artifacts",
        help="Directory where checkpoints and evaluation artifacts are stored.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=None,
        help=(
            "Resume training from the last checkpoint in --output-dir when passed as a flag, "
            "or from the explicit checkpoint path when a value is provided."
        ),
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
        "--logging-dir",
        default=None,
        help=(
            "Directory where framework-specific loggers (e.g., TensorBoard) will write events. "
            "Defaults to <output-dir>/runs when TensorBoard logging is enabled."
        ),
    )
    parser.add_argument(
        "--iteration-idle-time",
        type=parse_duration_to_seconds,
        default=0.0,
        help=(
            "Duration to sleep between training steps every --iteration-idle-interval iterations. "
            "Accepts values like '0.5s' or '250ms'."
        ),
    )
    parser.add_argument(
        "--iteration-idle-interval",
        type=int,
        default=0,
        help=(
            "Interval (in steps) for pausing during training/evaluation. Set alongside --iteration-idle-time."
        ),
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
        "--report-to",
        nargs="+",
        default=["tensorboard"],
        help=(
            "Integration targets for Hugging Face Trainer logging. Use 'tensorboard' to enable TensorBoard "
            "logging (requires the tensorboard package) or 'none' to disable external logging."
        ),
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help=(
            "If set, caps the number of training samples processed per epoch; "
            "each epoch draws a fresh random subset."
        ),
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
    parser.add_argument(
        "--sample-inspection-mode",
        choices=["off", "debug_path", "tensorboard"],
        default="debug_path",
        help=(
            "Control how good/bad training samples are dropped for inspection. "
            "Use 'debug_path' (default) to write artifacts to disk, 'tensorboard' to log them "
            "to TensorBoard, or 'off' to disable the feature."
        ),
    )
    parser.add_argument(
        "--sample-inspection-dir",
        default=None,
        help=(
            "Directory where inspection artifacts (texts, metrics, attention visualizations) "
            "are stored when --sample-inspection-mode=debug_path. Defaults to <output-dir>/debug_samples."
        ),
    )
    parser.add_argument(
        "--tracked-sample-index",
        type=int,
        default=None,
        help=(
            "Index of the evaluation sample to snapshot after each evaluation step. "
            "Must refer to the evaluation split after any --max-eval-samples truncation."
        ),
    )
    parser.add_argument(
        "--tracked-sample-id-column",
        default=None,
        help=(
            "Name of the dataset column used to locate the tracked evaluation sample. "
            "Requires --tracked-sample-id-value."
        ),
    )
    parser.add_argument(
        "--tracked-sample-id-value",
        default=None,
        help=(
            "Value in --tracked-sample-id-column for the evaluation sample to snapshot. "
            "Cannot be combined with --tracked-sample-index."
        ),
    )
    return parser


def parse_training_args(argv: Optional[Sequence[str]] = None) -> TrainingConfig:
    parser = build_arg_parser()
    namespace = parser.parse_args(argv)
    if namespace.tracked_sample_index is not None and namespace.tracked_sample_index < 0:
        parser.error("--tracked-sample-index must be a non-negative integer.")
    has_id_column = namespace.tracked_sample_id_column is not None
    has_id_value = namespace.tracked_sample_id_value is not None
    if has_id_column != has_id_value:
        parser.error("--tracked-sample-id-column requires --tracked-sample-id-value (and vice versa).")
    if namespace.tracked_sample_index is not None and has_id_column:
        parser.error("Use either --tracked-sample-index or --tracked-sample-id-*, not both.")
    if namespace.val_max_target_length is None:
        namespace.val_max_target_length = namespace.max_target_length
    namespace.report_to = normalize_report_to(namespace.report_to)
    return TrainingConfig(
        dataset_name=namespace.dataset_name,
        dataset_config=namespace.dataset_config,
        text_column=namespace.text_column,
        summary_column=namespace.summary_column,
        model_name=namespace.model_name,
        output_dir=namespace.output_dir,
        resume_from_checkpoint=namespace.resume_from_checkpoint,
        max_source_length=namespace.max_source_length,
        max_target_length=namespace.max_target_length,
        val_max_target_length=namespace.val_max_target_length,
        per_device_train_batch_size=namespace.per_device_train_batch_size,
        per_device_eval_batch_size=namespace.per_device_eval_batch_size,
        num_train_epochs=namespace.num_train_epochs,
        learning_rate=namespace.learning_rate,
        weight_decay=namespace.weight_decay,
        warmup_steps=namespace.warmup_steps,
        logging_steps=namespace.logging_steps,
        logging_dir=namespace.logging_dir,
        iteration_idle_time=namespace.iteration_idle_time,
        iteration_idle_interval=namespace.iteration_idle_interval,
        evaluation_strategy=namespace.evaluation_strategy,
        save_strategy=namespace.save_strategy,
        gradient_accumulation_steps=namespace.gradient_accumulation_steps,
        report_to=list(namespace.report_to),
        max_train_samples=namespace.max_train_samples,
        max_eval_samples=namespace.max_eval_samples,
        seed=namespace.seed,
        predict_with_generate=namespace.predict_with_generate,
        generation_num_beams=namespace.generation_num_beams,
        baseline_sentences=namespace.baseline_sentences,
        num_samples_for_report=namespace.num_samples_for_report,
        sample_inspection_mode=namespace.sample_inspection_mode,
        sample_inspection_dir=namespace.sample_inspection_dir,
        tracked_sample_index=namespace.tracked_sample_index,
        tracked_sample_id_column=namespace.tracked_sample_id_column,
        tracked_sample_id_value=namespace.tracked_sample_id_value,
    )
