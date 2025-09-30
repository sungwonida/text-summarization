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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from contextlib import contextmanager

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
import evaluate
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

LOGGER = logging.getLogger(__name__)


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

    args = parser.parse_args()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length
    args.report_to = normalize_report_to(args.report_to)
    return args


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


def _convert_metric_value(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return float(value.item())
    if isinstance(value, (int, float)):
        return float(value)
    return value


def compute_sample_quality_metrics(
    rouge_metric,
    bertscore_metric,
    article: str,
    reference_summary: str,
    predicted_summary: str,
    baseline_sentences: int,
):
    lead_summary = lead_n(article, baseline_sentences)

    lead_scores = rouge_metric.compute(
        predictions=[lead_summary],
        references=[reference_summary],
        use_stemmer=True,
    )
    lead_scores = {key: _convert_metric_value(value) for key, value in lead_scores.items()}

    rouge_scores = rouge_metric.compute(
        predictions=[predicted_summary],
        references=[reference_summary],
        use_stemmer=True,
    )
    rouge_scores = {key: _convert_metric_value(value) for key, value in rouge_scores.items()}

    bert_scores: Optional[Dict[str, float]] = None
    if bertscore_metric is not None:
        try:
            bertscore_result = bertscore_metric.compute(
                predictions=[predicted_summary],
                references=[reference_summary],
                lang="en",
            )
            bert_scores = {
                "precision": _convert_metric_value(bertscore_result["precision"][0]),
                "recall": _convert_metric_value(bertscore_result["recall"][0]),
                "f1": _convert_metric_value(bertscore_result["f1"][0]),
                "hashcode": bertscore_result.get("hashcode"),
            }
        except Exception as exc:  # pragma: no cover - defensive guard around optional dependency
            LOGGER.warning("Unable to compute BERTScore for inspection sample: %s", exc)

    metrics = {
        "lead3": lead_scores,
        "rouge": rouge_scores,
        "bertscore": bert_scores,
    }

    return lead_summary, metrics


def select_primary_score(metrics: Dict[str, Dict[str, float] | None]) -> float:
    bert_scores = metrics.get("bertscore")
    if isinstance(bert_scores, dict) and bert_scores.get("f1") is not None:
        return float(bert_scores["f1"])
    rouge_scores = metrics.get("rouge") or {}
    for key in ("rougeLsum", "rougeL", "rouge1", "rouge2"):
        value = rouge_scores.get(key)
        if value is not None:
            return float(value)
    return 0.0


def extract_attention_details(
    model,
    tokenizer,
    encoded_inputs,
    device_inputs,
    generated_ids,
):
    if generated_ids.size(-1) <= 1:
        return None

    decoder_input_ids = generated_ids[:, :-1]
    try:
        with torch.no_grad():
            outputs = model(
                **device_inputs,
                decoder_input_ids=decoder_input_ids,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
    except Exception as exc:  # pragma: no cover - guard for model incompatibilities
        LOGGER.warning("Unable to retrieve attention weights for inspection sample: %s", exc)
        return None

    cross_attentions = getattr(outputs, "cross_attentions", None)
    if not cross_attentions:
        return None

    try:
        attention_stack = torch.stack([layer[0] for layer in cross_attentions], dim=0)
        attention_matrix = attention_stack.mean(dim=1).mean(dim=0)
    except Exception as exc:  # pragma: no cover - guard for unexpected tensor shapes
        LOGGER.warning("Failed to aggregate attention weights: %s", exc)
        return None

    if "attention_mask" in encoded_inputs:
        input_len = int(encoded_inputs["attention_mask"][0].sum().item())
    else:
        input_len = encoded_inputs["input_ids"].shape[-1]

    output_len = attention_matrix.size(0)
    attention_matrix = attention_matrix[:output_len, :input_len].detach().cpu().numpy()

    input_tokens = tokenizer.convert_ids_to_tokens(
        encoded_inputs["input_ids"][0][:input_len].tolist()
    )
    decoder_tokens = tokenizer.convert_ids_to_tokens(
        decoder_input_ids[0][:output_len].detach().cpu().tolist()
    )

    return {
        "matrix": attention_matrix,
        "input_tokens": input_tokens,
        "output_tokens": decoder_tokens,
    }


def render_attention_heatmap(attention_details: Dict[str, object], title: str):
    matrix = np.asarray(attention_details["matrix"], dtype=np.float32)
    input_tokens = list(attention_details["input_tokens"])
    output_tokens = list(attention_details["output_tokens"])

    width = max(6.0, min(12.0, len(input_tokens) * 0.35))
    height = max(4.0, min(10.0, len(output_tokens) * 0.35))
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("Output Tokens")
    ax.set_title(title)

    ax.set_xticks(range(len(input_tokens)))
    ax.set_xticklabels(input_tokens, rotation=90, fontsize=6)
    ax.set_yticks(range(len(output_tokens)))
    ax.set_yticklabels(output_tokens, fontsize=6)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def figure_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape((height, width, 3))
    return image


def save_inspection_artifacts(
    sample: Optional[Dict[str, object]],
    tag: str,
    base_dir: Optional[Path],
    global_step: int,
    summary_writer,
    mode: str,
):
    if sample is None:
        return

    directory_basename = f"{tag}_sample"
    sample_dir: Optional[Path] = None
    if base_dir is not None:
        sample_dir = base_dir / f"step_{global_step:06d}" / directory_basename
        sample_dir.mkdir(parents=True, exist_ok=True)

    text_content_lines = [
        f"Sample type: {tag}",
        f"Sample index: {sample.get('sample_index')}",
        "",
        "Article:",
        str(sample.get("article", "")),
        "",
        "Reference Summary:",
        str(sample.get("reference_summary", "")),
        "",
        "Model Summary:",
        str(sample.get("model_summary", "")),
        "",
        "Lead Summary:",
        str(sample.get("lead_summary", "")),
        "",
        "Metrics:",
        json.dumps(sample.get("metrics", {}), indent=2),
    ]
    text_content = "\n".join(text_content_lines)

    if sample_dir is not None:
        text_path = sample_dir / f"{directory_basename}.txt"
        with text_path.open("w", encoding="utf-8") as f:
            f.write(text_content)

        metrics_path = sample_dir / f"{directory_basename}.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(sample.get("metrics", {}), f, indent=2)

    if summary_writer is not None and mode == "tensorboard":
        summary_writer.add_text(
            f"inspection/{tag}",
            text_content,
            global_step=global_step,
        )

    attention_details = sample.get("attention") if isinstance(sample, dict) else None
    if attention_details is None:
        return

    attention_title = f"{tag.title()} Sample Attention"
    fig = render_attention_heatmap(attention_details, attention_title)

    if sample_dir is not None:
        attention_path = sample_dir / f"{directory_basename}.png"
        fig.savefig(attention_path, bbox_inches="tight")

    if summary_writer is not None and mode == "tensorboard":
        image_array = figure_to_array(fig)
        summary_writer.add_image(
            f"inspection/{tag}/attention",
            image_array,
            global_step=global_step,
            dataformats="HWC",
        )

    plt.close(fig)


def build_training_arguments(args: argparse.Namespace, output_dir: Path) -> Seq2SeqTrainingArguments:
    """Create ``Seq2SeqTrainingArguments`` while remaining compatible with multiple versions."""

    base_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": args.resume_from_checkpoint is None,
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
        "report_to": args.report_to,
        "seed": args.seed,
    }

    if args.logging_dir is not None:
        base_kwargs["logging_dir"] = args.logging_dir

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


def normalize_report_to(report_to: Iterable[str]) -> List[str]:
    normalized = [entry.lower() for entry in report_to if entry]
    if "none" in normalized and len(normalized) > 1:
        LOGGER.warning("Ignoring 'none' in --report-to because other integrations were provided.")
        normalized = [entry for entry in normalized if entry != "none"]
    if not normalized:
        return ["none"]
    return normalized


def should_enable_tensorboard(report_to: Iterable[str]) -> bool:
    return any(entry == "tensorboard" for entry in report_to)


@contextmanager
def tensorboard_writer_context(args: argparse.Namespace, output_dir: Path):
    if not should_enable_tensorboard(args.report_to):
        yield None
        return

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "TensorBoard logging requested but tensorboard is not installed. Install it or pass --report-to none."
        ) from exc

    log_dir = Path(args.logging_dir) if args.logging_dir else output_dir / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("TensorBoard logs will be written to %s", log_dir)
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        yield writer
    finally:
        writer.flush()
        writer.close()

  
def main():
    configure_logging()
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_argument = args.resume_from_checkpoint
    resume_path: str | None = None
    last_checkpoint: str | None = None

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

    if should_enable_tensorboard(args.report_to):
        log_dir = Path(args.logging_dir) if args.logging_dir else output_dir / "runs"
        args.logging_dir = str(log_dir)
    elif args.logging_dir is not None:
        args.logging_dir = str(Path(args.logging_dir))

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


    with tensorboard_writer_context(args, output_dir) as summary_writer:
        inspection_mode = args.sample_inspection_mode
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
                    Path(args.sample_inspection_dir)
                    if args.sample_inspection_dir
                    else output_dir / "debug_samples"
                )
                base_dir.mkdir(parents=True, exist_ok=True)
                inspection_dir = base_dir
            elif args.sample_inspection_dir:
                inspection_dir = Path(args.sample_inspection_dir)
                inspection_dir.mkdir(parents=True, exist_ok=True)

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

        if args.iteration_idle_time > 0 and args.iteration_idle_interval > 0:
            trainer.add_callback(
                IterationPauseCallback(
                    pause_seconds=args.iteration_idle_time,
                    interval=args.iteration_idle_interval,
                )
            )

        if summary_writer is not None:
            summary_writer.add_text(
                "config/dataset",
                json.dumps(
                    {
                        "name": args.dataset_name,
                        "config": args.dataset_config,
                        "text_column": columns.text_column,
                        "summary_column": columns.summary_column,
                        "max_train_samples": args.max_train_samples,
                        "max_eval_samples": args.max_eval_samples,
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

        qualitative_samples: List[Dict[str, object]] = []
        best_inspection_sample: Optional[Dict[str, object]] = None
        worst_inspection_sample: Optional[Dict[str, object]] = None
        bertscore_metric = None
        if inspection_enabled:
            try:
                bertscore_metric = evaluate.load("bertscore")
            except Exception as exc:  # pragma: no cover - network/model download issues
                LOGGER.warning("Unable to load BERTScore metric for sample inspection: %s", exc)
                bertscore_metric = None

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
                    encoded_inputs = tokenizer(
                        input_text,
                        return_tensors="pt",
                        max_length=args.max_source_length,
                        truncation=True,
                    )
                    device_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
                    with torch.no_grad():
                        generation = model.generate(
                            **device_inputs,
                            max_length=args.val_max_target_length,
                            num_beams=args.generation_num_beams,
                            return_dict_in_generate=True,
                        )
                    generated_ids = generation.sequences
                    predicted_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                    lead_summary = lead_n(input_text, args.baseline_sentences)
                    metrics = None
                    primary_score = None
                    attention_details = None

                    if inspection_enabled:
                        attention_details = extract_attention_details(
                            model,
                            tokenizer,
                            encoded_inputs,
                            device_inputs,
                            generated_ids,
                        )
                        lead_summary, metrics = compute_sample_quality_metrics(
                            rouge_metric=rouge,
                            bertscore_metric=bertscore_metric,
                            article=input_text,
                            reference_summary=reference_summary,
                            predicted_summary=predicted_summary,
                            baseline_sentences=args.baseline_sentences,
                        )
                        primary_score = select_primary_score(metrics)

                    qualitative_sample: Dict[str, object] = {
                        "article": input_text,
                        "reference_summary": reference_summary,
                        "model_summary": predicted_summary,
                        "lead_summary": lead_summary,
                    }
                    if metrics is not None:
                        qualitative_sample["metrics"] = metrics
                    qualitative_samples.append(qualitative_sample)

                    if inspection_enabled and metrics is not None:
                        inspection_record = {
                            "article": input_text,
                            "reference_summary": reference_summary,
                            "model_summary": predicted_summary,
                            "lead_summary": lead_summary,
                            "metrics": metrics,
                            "attention": attention_details,
                            "sample_index": idx,
                            "primary_score": primary_score,
                        }
                        if (
                            best_inspection_sample is None
                            or primary_score > best_inspection_sample.get("primary_score", float("-inf"))
                        ):
                            best_inspection_sample = inspection_record
                        if (
                            worst_inspection_sample is None
                            or primary_score < worst_inspection_sample.get("primary_score", float("inf"))
                        ):
                            worst_inspection_sample = inspection_record

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

            inspection_summary = {
                "mode": inspection_mode,
                "good": serialize_inspection_sample(best_inspection_sample),
                "bad": serialize_inspection_sample(worst_inspection_sample),
            }

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

        if inspection_summary:
            report["inspection"] = inspection_summary

        report_path = output_dir / "evaluation_report.json"
        LOGGER.info("Writing evaluation report to %s", report_path)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
