"""Utilities for qualitative inspection and attention visualization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import TrainingConfig
from .datasets import ColumnMapping, lead_n

LOGGER = logging.getLogger(__name__)


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
    if isinstance(bert_scores, dict):
        f1_score = bert_scores.get("f1")
        if f1_score is not None:
            return float(f1_score)
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

    input_tokens = tokenizer.convert_ids_to_tokens(encoded_inputs["input_ids"][0][:input_len].tolist())
    decoder_tokens = tokenizer.convert_ids_to_tokens(decoder_input_ids[0][:output_len].detach().cpu().tolist())

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


def collect_samples_for_inspection(
    dataset,
    *,
    columns: ColumnMapping,
    model,
    tokenizer,
    device: torch.device,
    config: TrainingConfig,
    rouge_metric,
    bertscore_metric,
    inspection_enabled: bool,
):
    qualitative_samples: List[Dict[str, object]] = []
    best_sample: Optional[Dict[str, object]] = None
    worst_sample: Optional[Dict[str, object]] = None

    if dataset is None:
        return qualitative_samples, best_sample, worst_sample

    total_records = len(dataset)
    if total_records == 0:
        return qualitative_samples, best_sample, worst_sample

    max_samples = min(config.num_samples_for_report, total_records)
    sample_indices = range(max_samples)

    for idx in sample_indices:
        record = dataset[idx]
        input_text = record[columns.text_column]
        reference_summary = record[columns.summary_column]

        encoded_inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=config.max_source_length,
            truncation=True,
        )
        device_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}

        with torch.no_grad():
            generation = model.generate(
                **device_inputs,
                max_length=config.val_max_target_length,
                num_beams=config.generation_num_beams,
                return_dict_in_generate=True,
            )

        generated_ids = generation.sequences
        predicted_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        attention_details = None
        lead_summary = lead_n(input_text, config.baseline_sentences)
        metrics = None
        primary_score = None

        if inspection_enabled:
            attention_details = extract_attention_details(
                model,
                tokenizer,
                encoded_inputs,
                device_inputs,
                generated_ids,
            )
            lead_summary, metrics = compute_sample_quality_metrics(
                rouge_metric=rouge_metric,
                bertscore_metric=bertscore_metric,
                article=input_text,
                reference_summary=reference_summary,
                predicted_summary=predicted_summary,
                baseline_sentences=config.baseline_sentences,
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
                best_sample is None
                or primary_score > best_sample.get("primary_score", float("-inf"))
            ):
                best_sample = inspection_record
            if (
                worst_sample is None
                or primary_score < worst_sample.get("primary_score", float("inf"))
            ):
                worst_sample = inspection_record

    return qualitative_samples, best_sample, worst_sample
