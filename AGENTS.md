# Repository Guidelines

## Project Structure & Module Organization
Core Python packages live in `summarizer/`, with modules split by responsibility: `config.py` centralizes dataclasses and CLI defaults, `datasets.py` handles dataset ingestion and column inference, `training.py` and `runtime.py` build Hugging Face Trainer objects, while `callbacks.py`, `logging_utils.py`, and `inspection.py` capture metrics and qualitative samples. Executable entry points reside in `scripts/`; `train_abstractive.py` is the baseline fine-tuning driver, `occlusion_loso.py` supports leave-one-sentence-out saliency checks, and `save_attention_maps.py` exports attention heatmaps. Experiment results land under `outputs/<run-name>/`; treat them as reproducible artifacts and keep evaluation reports when citing metrics.

## Build, Test, and Development Commands
Create an isolated environment with `python -m venv .venv`, `source .venv/bin/activate`, and `pip install -r requirements.txt`. Run the baseline loop via:
```bash
python scripts/train_abstractive.py --dataset-name cnn_dailymail --dataset-config 3.0.0 --predict-with-generate --output-dir outputs/abstractive
```
For quick smoke checks append `--max-train-samples 2000 --max-eval-samples 500`. Launch TensorBoard with `tensorboard --logdir outputs/<run-name>/runs` to monitor metrics and qualitative samples.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation, descriptive lowercase module names, and CapWords class names. Add type hints to public functions and reuse dataclass patterns from `summarizer/config.py`. Prefer pathlib objects for filesystem paths, keep configurable values in `TrainingConfig`, and expose new CLI options through the existing argument builders. When extending scripts, keep CLI flags kebab-cased to match Hugging Face conventions.

## Testing Guidelines
Automated unit tests are not yet in place; rely on controlled training and evaluation runs before merging. Execute the training script with reduced sample counts to verify data loading, metric computation, and checkpoint writes. Inspect the generated `evaluation_report.json` for ROUGE fields and qualitative samples, and note any regressions. Capture relevant artifacts (ROUGE deltas, sample summaries, attention maps) when experimentation changes behavior.

## Commit & Pull Request Guidelines
Commit messages use imperative mood with optional scope prefixes (e.g., `refactor(core): ...`, `chore: ...`). Group related edits into a single commit that mirrors one rationale. Pull requests should summarize motivation, list dataset or configuration changes, and link issues or experiment IDs. Include reproduction notes (exact commands, key flags) and reference any new files added under `outputs/`. Add screenshots or JSON snippets when they help reviewers assess qualitative shifts.
