# text-summarization

This repository hosts a configurable abstractive summarization pipeline built with
PyTorch and Hugging Face Transformers. The project focuses on fine-tuning a
pretrained seq2seq model on English news articles, evaluating it with ROUGE, and
capturing qualitative artifacts that make experimentation and comparison easy.

## Abstractive Summarization Baseline

The baseline training flow fine-tunes `sshleifer/distilbart-cnn-12-6` on
`cnn_dailymail` (customizable via CLI flags) and provides:

- Command-line driven configuration that works across CPU, CUDA, and Apple Silicon
  (`mps`) devices.
- Automatic dataset column inference for common summarization datasets.
- ROUGE-based evaluation alongside an extractive lead-N baseline for context.
- JSON evaluation reports containing metrics, configuration details, and sample
  summaries.
- Optional TensorBoard logging for metrics and qualitative examples.

### Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are on Apple Silicon and want to leverage the GPU, ensure that PyTorch is
installed with MPS support (`pip install torch --index-url https://download.pytorch.org/whl/cpu`
for CPU-only, or follow the [official instructions](https://pytorch.org/get-started/locally/)
for GPU-enabled builds).

### Training & evaluation

```bash
python scripts/train_abstractive.py \
  --dataset-name cnn_dailymail \
  --dataset-config 3.0.0 \
  --predict-with-generate \
  --output-dir artifacts
```

The command downloads the dataset, fine-tunes the baseline model, evaluates it,
and produces an `evaluation_report.json` file inside `artifacts/`.
Use `--max-train-samples` and `--max-eval-samples` for quick smoke tests on
limited hardware (for example, `--max-train-samples 2000 --max-eval-samples 500`).

To resume a previous run, pass `--resume-from-checkpoint` to load the most recent
checkpoint in `--output-dir`, or provide an explicit path such as
`--resume-from-checkpoint artifacts/checkpoint-1000`.

### TensorBoard logging

TensorBoard logging is enabled by default through Hugging Face's Trainer
integration. Run TensorBoard in a separate shell to monitor loss curves, ROUGE
scores, and qualitative samples:

```bash
tensorboard --logdir artifacts/runs
```

Use `--logging-dir` to customize the event directory and `--report-to none` to
disable the integration entirely.

### Output artifacts

- `artifacts/evaluation_report.json` – captures ROUGE scores, lead-N
  baseline metrics, CLI arguments, and qualitative examples.
- Model checkpoints and tokenizer files saved alongside the report, ready for
  inference or further fine-tuning.

### Reproducing qualitative samples

The report contains a short list of model vs. reference summaries taken from the
evaluation split. Increase `--num-samples-for-report` for a larger qualitative
slice, or inspect individual samples programmatically from the JSON file.
