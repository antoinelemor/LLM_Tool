# LLM Tool Mode Playbook

LLM Tool packs six interactive modes that can be combined into a full research pipeline.  
This playbook explains what each mode does, what it expects, and how it feeds the next step.  
The structure is identical across modes so you can compare them quickly.

```
┌───────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Mode 1       │──▶──│  Mode 2           │──▶──│  Mode 3        │
│  Annotator    │     │  Annotator Factory│     │  Training Arena│
└───────────────┘     └────────┬─────────┘     └───────┬───────┘
                                │                     │
                                ▼                     ▼
                         ┌───────────────┐     ┌───────────────┐
                         │  Mode 4       │     │  Mode 5       │
                         │  BERT Studio  │     │  Validation   │
                         └───────────────┘     └───────────────┘
                                     ▲                 │
                                     └───────┬─────────┘
                                             ▼
                                   ┌──────────────────┐
                                   │  Mode 6          │
                                   │  Profile Manager │
                                   └──────────────────┘
```

---

## Mode 1 – The Annotator

- **Mission**: Generate first-pass annotations with cloud or local large language models, enforcing schema consistency and incremental checkpoints.
- **Best For**: Rapid coding of raw qualitative data, exploring category schemes, creating seed datasets before human review.
- **Inputs**: CSV, TSV, Excel (`.xlsx`/`.xls`), Parquet, JSON/JSONL, PostgreSQL tables, RData/RDS; prompt definitions from files, the prompt wizard, or prior sessions.
- **Outputs**: Annotated tables (`annotations_output/<session>/data/*.csv|jsonl`), cleaned prompts, per-prompt JSON traces, resume metadata, optional Label Studio/Doccano exports, quality summaries.
- **Core Steps**:
  1. Detect candidate datasets (auto-scans folders and previews schema).
  2. Select text/ID columns and optional metadata fields.
  3. Pick an LLM provider (OpenAI, Anthropic, Google, Ollama, LlamaCPP) and model.
  4. Design prompts (simple, multi-prompt, or guided Social Science Wizard).
  5. Set execution controls (batch size, retries, confidence sampling, JSON validation).
  6. Launch annotation with live Rich dashboard (success/failure counters, throughput).
  7. Save exports and optional validation samples for Mode 5.
- **Automation Highlights**:
  - 5-stage JSON repair loop with schema validation and type coercion.
  - Sample-size estimator (95% CI) to suggest human review workload.
  - Incremental writes so you can pause/resume without data loss.
  - Built-in API key vault with Fernet encryption and environment-variable fallback.
- **Resume & Logging**:
  - Session metadata in `logs/annotator/<session>/resume.json`.
  - Detailed run log in `logs/annotator/<session>/annotator.log`.
  - Prompt variants stored under `annotations_output/<session>/prompts/`.
- **Hand-off & Next Mode**:
  - Send cleaned annotations to Mode 5 for validation or Mode 2 to continue into training.
  - Export JSONL to Label Studio/Doccano for collaborative human coding.

---

## Mode 2 – The Annotator Factory

- **Mission**: Orchestrate annotation, dataset preparation, stratified splitting, and training launch as one reproducible pipeline.
- **Best For**: Teams that want an auditable, end-to-end workflow from raw text to deployable models.
- **Inputs**: Any dataset supported by Mode 1 plus prior Annotator session metadata, optional validation benchmarks, and configuration profiles.
- **Outputs**: Normalised training corpora, split manifests, benchmark reports, inference-ready models, factory session logs.
- **Core Steps**:
  1. Import or rerun an Annotator session (supports warm start with stored prompts).
  2. Analyse class distribution, languages, missing data, and quality flags.
  3. Generate training bundles with `TrainingDatasetBuilder` (single, multi-label, JSON).
  4. Configure stratified splits, hold-out strategy, and validation plans.
  5. Launch Mode 3 from inside the pipeline with prepared configuration payloads.
  6. Optionally chain into Mode 4 for immediate model inference against new data.
  7. Archive the entire pipeline artefacts in `logs/annotator_factory/<session>/`.
- **Automation Highlights**:
  - Heuristic column detection (text, labels, metadata) via `DataDetector`.
  - Language-aware balancing and minority-class boosting.
  - Session-wide JSON timeline for reruns and provenance.
  - Rich summary dashboards with throughput charts and status badges.
- **Resume & Logging**:
  - Resume maps stored at `logs/annotator_factory/<session>/resume.json`.
  - Training/prep metrics in `models/<session>/metrics/` and `logs/application/`.
- **Hand-off & Next Mode**:
  - Launches Mode 3 (Training Arena) and Mode 4 (BERT Annotation Studio) automatically.
  - Provides structured exports for manual inspection or external MLOps pipelines.

---

## Mode 3 – Training Arena

- **Mission**: Train, benchmark, and select the best transformer models (50+ architectures) for your annotated corpus.
- **Best For**: Researchers who want multilingual, multi-label, or benchmarked models without writing PyTorch code.
- **Inputs**: Training bundles from Mode 2 or externally prepared datasets (CSV/JSONL), configuration profiles, language hints.
- **Outputs**: Hugging Face-style checkpoints, comparison tables, confusion matrices, training curves, performance summaries.
- **Core Steps**:
  1. Inspect dataset health (text length, label balance, multilingual coverage).
  2. Choose training strategy (single-label, multi-label, multi-class groups).
  3. Auto-select candidate models per language or override manually.
  4. Configure epochs, batch sizes, warm-up, and reinforcement passes.
  5. Train sequentially or in parallel with live progress bars and GPU stats.
  6. Review metrics, pick best-performing checkpoints, and export summaries.
  7. Optionally trigger Mode 4 to put the chosen model into production annotation.
- **Automation Highlights**:
  - Multilingual-aware selector (CamemBERT, XLM-R, DeBERTa, Longformer, etc.).
  - Reinforced learning loop to boost underperforming classes.
  - Auto FP16 / gradient accumulation recommendations based on system resources.
  - Metrics persisted to `models/<session>/metrics/*.json` with HTML dashboards.
- **Resume & Logging**:
  - Session state saved in `logs/training_arena/<session>/resume.json`.
  - Trainer logs under `models/<session>/training_logs/`.
- **Hand-off & Next Mode**:
  - Exports checkpoints to Mode 4 (BERT Annotation Studio) for inference.
  - Shares metrics with Mode 5 to inform validation sampling strategies.

---

## Mode 4 – BERT Annotation Studio

- **Mission**: Run trained transformer models in production-style annotation workflows with rich monitoring and export options.
- **Best For**: High-volume inference, cascaded model pipelines, or deploying models back to research assistants.
- **Inputs**: Checkpoints from Mode 3 (or custom Hugging Face models), fresh datasets, column mapping rules, optional cascading logic.
- **Outputs**: Annotated datasets, per-model confidence trails, audit-ready session reports, incremental cache files.
- **Core Steps**:
  1. Start or resume an annotation studio session (`logs/annotation_studio/<session>/`).
  2. Select one or many models; define cascading or consensus rules.
  3. Map incoming dataset columns (text, IDs, metadata) and language handling.
  4. Configure batch size, parallelism, cache reuse, and fallback strategies.
  5. Launch inference with live progress, confidence histograms, and sample previews.
  6. Export results to CSV/JSON/Label Studio/Doccano and update reporting dashboards.
  7. Close the session with a consolidated summary and next-step prompts.
- **Automation Highlights**:
  - Session manager keeps step-by-step cache, enabling exact restarts.
  - Language detector gates models to compliant passages automatically.
  - Parallel inference engine uses CPU, CUDA, or Apple MPS seamlessly.
  - Built-in text cleaning hooks and metadata propagation.
- **Resume & Logging**:
  - Sessions tracked by `AnnotationStudioSessionManager` with per-step metadata.
  - Artefacts stored in `logs/annotation_studio/<session>/artifacts/`.
- **Hand-off & Next Mode**:
  - Feed Mode 5 for spot-check validation or export predictions to downstream systems.
  - Update profiles in Mode 6 for future reproducibility.

---

## Mode 5 – Validation Lab

- **Mission**: Audit annotation quality, compute agreement metrics, and assemble human-review packages.
- **Best For**: Quality assurance, publication audit trails, IRB-compliant verification, inter-annotator studies.
- **Inputs**: Annotated datasets from Modes 1 or 4, human-coded benchmarks, configuration files specifying sampling strategies.
- **Outputs**: Validation subsets, agreement scores, confidence diagnostics, Doccano/Label Studio review sets, issue logs.
- **Core Steps**:
  1. Load annotations (`csv`, `json`, `jsonl`, `parquet`), including multiple annotator columns.
  2. Validate schema integrity and detect missing or inconsistent labels.
  3. Sample items (random or stratified) based on target sample sizes.
  4. Compute Cohen’s Kappa, accuracy, confusion matrices, and label balance.
  5. Flag low-confidence or conflicting items and export review packs.
  6. Document issues in run reports for traceability.
  7. Hand feedback to Modes 1–4 for re-annotation or retraining.
- **Automation Highlights**:
  - Auto-identifies label columns (supports `label`, `annotations`, multi-annotator fields).
  - Maintains provenance metadata per sampled item (source row, prompt ID, model).
  - Produces both machine-readable (`.jsonl`) and human-friendly (`.csv`) review bundles.
- **Resume & Logging**:
  - Validation configs stored beside outputs under `logs/validation/<timestamp>/`.
  - Issue summaries captured in `logs/application/llmtool_*.log`.
- **Hand-off & Next Mode**:
  - Supplies curated datasets back to Mode 1 for correction or Mode 3 for retraining.
  - Generates documentation packages suitable for appendices or replication kits.

---

## Mode 6 – Profile Manager

- **Mission**: Centralise credentials, default settings, and reusable prompt/model presets for reproducible experiments.
- **Best For**: Multi-provider setups, teams sharing machines, researchers alternating between cloud and local runs.
- **Inputs**: API keys, preferred models per provider, saved prompt bundles, execution history.
- **Outputs**: Encrypted key store (`~/.llm_tool/api_keys.enc`), profile JSON files, history logs, reusable configurations surfaced in the CLI.
- **Core Steps**:
  1. Open Profile Manager from the main menu (option 6).
  2. Review existing profiles and activate one as the default context.
  3. Add or update provider credentials with encryption (Fernet-backed).
  4. Save prompt presets, schema templates, and annotation settings for later reuse.
  5. Inspect execution history to reopen past sessions from Modes 1–4.
  6. Export or purge profiles when rotating machines or complying with data policies.
- **Automation Highlights**:
  - Environment-variable override detection for CI/CD or shared servers.
  - Per-provider preferred model tracking (e.g., default `gpt-4o` vs `ollama:llama3.2`).
  - Profile snapshots referenced automatically by advanced CLI wizards.
- **Resume & Logging**:
  - Profiles stored in `~/.llm_tool/profiles/`, history in `~/.llm_tool/history.json`.
  - All changes recorded in application logs with masked secrets.
- **Hand-off & Next Mode**:
  - Seeds credentials for Modes 1–5.
  - Ensures reproducible settings when returning to a project weeks or months later.

---

### Need Another View?

- Interactive CLI banners (Mode 7 in the menu) surface condensed help with the same structure.
- For API key specifics see `docs/API_KEY_MANAGEMENT.md`.
- Wizard walkthrough examples live in `docs/social_science_wizard_guide.md` and `docs/wizard_example_outputs.md`.

