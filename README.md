# text-summarization

This repository hosts the hands-on deliverables for Weeks 3 and 4 of the 12-week NLP journey.  The initial focus is on building a **baseline abstractive summarizer for English news articles** (Week 3) using PyTorch and Hugging Face Transformers.  Week 4 will extend the work to multilingual summarization once the English pipeline is solidified.

## Week 3 – Abstractive Summarization Baseline

The Week 3 goal is to fine-tune a sequence-to-sequence transformer (default: `sshleifer/distilbart-cnn-12-6`) on an English summarization dataset (default: `cnn_dailymail`).  The baseline pipeline includes:

- Configurable training script with support for Apple Silicon `mps`, NVIDIA `cuda`, or CPU devices.
- Automatic dataset column inference for common summarization datasets.
- ROUGE-based evaluation through Hugging Face `evaluate` and an extractive **lead-3 baseline** for comparison.
- JSON evaluation report capturing metrics, configuration, and qualitative samples.
- Optional TensorBoard logging for training/evaluation metrics and qualitative samples.

### Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are on Apple Silicon and want to leverage the GPU, ensure that PyTorch is installed with MPS support (`pip install torch --index-url https://download.pytorch.org/whl/cpu` for CPU-only, or follow the [official instructions](https://pytorch.org/get-started/locally/) for GPU-enabled builds).

### Training & evaluation

```bash
python week3/train_abstractive.py \
  --dataset-name cnn_dailymail \
  --dataset-config 3.0.0 \
  --predict-with-generate \
  --output-dir outputs/week3
```

The command above downloads the CNN/DailyMail dataset, fine-tunes the baseline model, evaluates it, and produces an `evaluation_report.json` file inside `outputs/week3/`.  Use `--max-train-samples` and `--max-eval-samples` for quick smoke tests on limited hardware (for example, `--max-train-samples 2000 --max-eval-samples 500`).

To resume a previous run, pass `--resume-from-checkpoint` to load the most recent checkpoint in `--output-dir`, or provide an explicit path such as `--resume-from-checkpoint outputs/week3/checkpoint-1000`.

### TensorBoard logging

TensorBoard logging is enabled by default through Hugging Face's Trainer integration. Run TensorBoard in a separate shell to monitor loss curves, ROUGE scores, and qualitative samples:

```bash
tensorboard --logdir outputs/week3/runs
```

Use `--logging-dir` to customize the event directory and `--report-to none` to disable the integration entirely.

### Output artifacts

- `outputs/week3/evaluation_report.json` – captures ROUGE scores, lead-n baseline metrics, CLI arguments, and qualitative examples.
- Model checkpoints and tokenizer files saved alongside the report, ready for inference or further fine-tuning.

### Reproducing qualitative samples

The report contains a short list of model vs. reference summaries taken from the evaluation split.  Increase `--num-samples-for-report` for a larger qualitative slice, or inspect individual samples programmatically from the JSON file.

---

Follow-up work for **Week 4 (multilingual summarization)** will build on this baseline to cover Korean and additional languages.
