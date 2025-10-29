"""Dataset utilities for summarization training and evaluation."""

from __future__ import annotations

import re
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import evaluate
import numpy as np
from datasets import DatasetDict

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


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
    *,
    seed: int | None = None,
) -> Dict[str, float]:
    metric = evaluate.load("rouge")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        metric.seed = seed
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


def preprocess_function(
    tokenizer,
    text_column: str,
    summary_column: str,
    max_source_length: int,
    max_target_length: int,
):
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
