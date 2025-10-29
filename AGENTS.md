# Agent Architecture

Agents in this project are composable services that coordinate data ingestion, model fine-tuning, evaluation, and qualitative analysis for abstractive summarization. Each agent exposes a narrow contract so the surrounding system can evolve—renaming modules or reshuffling directories should not affect the expectations between roles.

## Core Concepts
- **Agent** – an encapsulated unit that owns a responsibility (loading data, orchestrating training, publishing reports) and communicates via typed payloads rather than shared globals.
- **Session** – the lifecycle context in which agents collaborate; a session begins when configuration is parsed and ends after artifacts are persisted.
- **Payloads** – structured inputs and outputs passed between agents. Common payloads are configuration objects, tokenized datasets, trainer handles, metrics dictionaries, and qualitative samples.
- **Artifacts** – durable outputs (checkpoints, baselines, attention maps, evaluation reports) produced by agents for downstream consumption.

## Core Roles

### Orchestrator Agent
- **Responsibilities:** bootstrap logging, parse configuration, acquire devices, and wire the rest of the agents together. It owns the top-level session and guarantees that resources are created and released in a predictable order.
- **Inputs:** CLI arguments or programmatic overrides, environment hints (e.g., selected device), and optional experiment metadata.
- **Outputs:** consolidated run directory, final evaluation report, and a summary of qualitative findings.
- **Lifecycle:** configure → initialize dependencies → delegate work to data, training, and inspection agents → finalize artifacts.

### Data Agent
- **Responsibilities:** load raw datasets, infer column semantics, apply preprocessing/tokenization, and optionally compute extractive baselines for comparison.
- **Inputs:** dataset identifiers, optional column overrides, tokenizer handle, and configuration limits on sequence length or sample count.
- **Outputs:** tokenized dataset splits aligned with trainer expectations, column mappings, and baseline metric summaries.
- **Lifecycle:** resolve dataset → infer structure → build preprocessing transforms → emit tokenized payloads → supply optional baseline scores.

### Model & Training Agent
- **Responsibilities:** instantiate the seq2seq model, construct trainer arguments compatible with different library versions, register callbacks, and drive the training/evaluation loop.
- **Inputs:** configuration payloads, tokenized datasets, model checkpoint identifiers, and metric functions.
- **Outputs:** trained model weights, evaluation metrics, trainer state (for resuming), and emitted events for monitoring agents.
- **Lifecycle:** select device → load tokenizer/model → build training arguments → create trainer → run train/eval cycles → persist checkpoints.

### Monitoring & Inspection Agent
- **Responsibilities:** capture intermediate samples, compute per-example quality metrics (ROUGE, BERTScore, lead baselines), and optionally log to TensorBoard or structured debug directories.
- **Inputs:** trainer hooks, raw inspection dataset, model/tokenizer handles, configuration toggles for mode (“off”, debug path, or TensorBoard), and metric implementations.
- **Outputs:** qualitative sample collections, best/worst exemplars, serialized inspection artifacts, and summary statistics for dashboards.
- **Lifecycle:** hook into training callbacks → retrieve predictions at evaluation boundaries → compute metrics and select exemplars → persist artifacts → expose reusable metrics to the orchestrator.

### Research Diagnostics Agents
- **Responsibilities:** offer focused qualitative tools such as cross-attention visualization and leave-one-sentence-out occlusion analysis. These agents operate offline, consuming saved checkpoints and custom document lists.
- **Inputs:** trained checkpoints, raw documents (optionally paired with references), generation parameters, and plotting configuration.
- **Outputs:** heatmaps, occlusion rankings, and human-readable diagnostics that support model interpretation.
- **Lifecycle:** load model/tokenizer in evaluation mode → run targeted inference → transform signals into visual or textual artifacts.

### Reporting Agent
- **Responsibilities:** collate outputs from other agents into reproducible experiment artifacts (JSON reports, tensorboard summaries, optional markdown digests) suitable for sharing and regression tracking.
- **Inputs:** configuration snapshot, evaluation metrics, qualitative samples, and baseline comparisons.
- **Outputs:** structured reports, metadata for dashboards, and coordination with storage destinations supplied by the orchestrator.
- **Lifecycle:** normalize payloads → serialize to disk or monitoring sinks → signal completion back to the orchestrator.

## Interfaces and Lifecycle
1. **Configuration phase:** a typed configuration object encapsulates CLI options, environment overrides, and experiment metadata. Agents receive only the pieces they need, so renaming flags or relocating parsers does not ripple through the system.
2. **Preparation phase:** data and runtime agents allocate resources (datasets, devices, tokenizers) and return lightweight descriptors plus cleanup hooks.
3. **Execution phase:** the training agent coordinates the core loop while delegating observability to monitoring agents. Payloads exchanged during this phase include batches, trainer state, and callback events.
4. **Evaluation phase:** evaluation metrics, extractive baselines, and qualitative samples are produced in parallel, enabling the orchestrator to decide which artifacts to persist.
5. **Completion phase:** reporting agents write durable artifacts, inspection agents finalize their logs, and the orchestrator tears down any open sessions.

## Key Workflows
- **Baseline fine-tuning:** orchestrator parses configuration → data agent tokenizes splits → training agent builds the trainer and attaches iteration/inspection callbacks → monitoring agent records per-step metrics → reporting agent writes the evaluation report. Resume modes reuse the trainer state and checkpoint discovery logic without altering the flow.
- **Sample inspection:** during evaluation hooks, monitoring agents gather generated summaries, compute ROUGE/BERTScore, compare against lead baselines, surface best/worst examples, and log to TensorBoard or a debug directory for later review.
- **Qualitative diagnostics:** standalone agents load checkpoints post-training, generate attention maps or occlusion scores for researcher-supplied documents, and surface ranked findings to guide error analysis.
- **Experiment review:** reporting agents fold together configuration snapshots, metrics trends, and qualitative artifacts so subsequent runs can validate regressions or improvements without replaying the entire job.

## Extensibility and Conventions
- **Configuration growth:** introduce new flags by extending the configuration schema and ensuring the orchestrator distributes the new field to interested agents. Prefer typed durations (e.g., milliseconds parsing) and lower-case logging keywords, preserving the existing ergonomic style.
- **Dataset customizations:** support alternate corpora or column layouts by enhancing the data agent’s column inference rules or allowing explicit overrides. Heavy preprocessing should remain encapsulated to avoid leaking tokenization assumptions to other agents.
- **Training hooks:** compose additional callbacks (e.g., pacing controls, curriculum schedulers) through the training agent. Callbacks should respect the same logging conventions and avoid blocking the trainer thread unless explicitly configured.
- **Monitoring targets:** plug new metrics or visualization sinks into the monitoring agent by exposing lightweight adapters. Keep optional dependencies guarded so environments without certain libraries continue to run.
- **Reporting outputs:** when adding artifact types, register them with the reporting agent so the orchestrator can advertise the new deliverables without assuming a specific directory layout.
- **General conventions:** adhere to type hints, dataclasses for configuration payloads, and the standard logging framework; avoid printing directly from agents to keep orchestration deterministic.

## Implementation Examples
```python
def run_experiment(argv: list[str]) -> None:
    config = load_configuration(argv)
    with instrumentation_session(config) as session:
        datasets = data_agent.prepare(config, session.device, session.tokenizer)
        trainer = training_agent.create(config=config, datasets=datasets, session=session)
        trainer.train()
        metrics = trainer.evaluate()
        inspections = monitoring_agent.collect(trainer, datasets.validation)
        report_agent.persist(config=config, metrics=metrics, inspections=inspections)
```

```python
class NovelInspectionCallback(TrainerCallback):
    def __init__(self, inspector):
        self.inspector = inspector

    def on_evaluate(self, args, state, control, **kwargs):
        samples = self.inspector.collect(state.global_step)
        for sample in samples:
            publish_sample(sample)
        return control
```

These snippets illustrate how orchestration code composes agents without depending on concrete module paths; implementations can migrate across packages while preserving contracts.

## Evolution and Intent
- Early iterations focused on a reliable baseline trainer and compatibility across library versions.
- Subsequent updates introduced resumable checkpoints, TensorBoard logging, configurable iteration pacing, and structured logging to support long-running experiments.
- Later work layered on qualitative tooling—automatic sample inspection, cross-attention visualization, and occlusion-based diagnostics—highlighting the need for agents that can operate both inline (during training) and offline (post-hoc analysis).
- Maintaining these abstractions keeps “agent” expectations stable even as refactors reshape modules, enabling future contributors to extend functionality without re-discovering historical context.
