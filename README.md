# MRI_Agent v2.0

Agentic MRI analysis framework with a decoupled **LLM planner** and **deterministic execution engine**.
Designed for auditable, domain-constrained clinical imaging workflows across **prostate**, **brain**, and **cardiac** MRI.

## Why v2.0
MRI_Agent v2.0 moves beyond single-pass tool calling. It introduces a robust agent runtime with planning, guarded execution, reflection, and case-local QA over deep DICOM metadata.

Core outcomes:
- Stronger reliability under tool/runtime failures.
- Better safety boundaries for medical workflows.
- Better operator UX for interactive and scripted use.
- Better explainability with traceable artifacts and evidence-backed responses.

## Architecture

### Brain-Cerebellum Split
V2 formalizes a two-layer architecture:

- **Brain (LangGraph Planner)**
  - Interprets user intent.
  - Builds a dependency-aware DAG plan.
  - Supports request-type routing (`full_pipeline`, `segment`, `register`, `lesion`, `classify`, `qa`, `custom_analysis`).
  - Applies dynamic argument inference from natural language and case context.

- **Cerebellum (Deterministic Executor)**
  - Executes DAG nodes with strict contracts.
  - Validates preconditions, schema, and scope.
  - Persists auditable run state and traces.
  - Enforces predictable tool semantics independent of LLM variability.

This separation is intentional: the LLM decides *what to do*, the deterministic runtime decides *how it is safely done*.

### Reflector Engine (Self-Healing Loop)
The runtime includes an active reflection/debug cycle:

- Detects failed, blocked, or malformed tool calls.
- Uses structured run context to propose safe argument repairs.
- Retries with corrected parameters under policy controls.
- Preserves full trace history for post-hoc audit.

This makes execution resilient to common planner errors without allowing uncontrolled behavior.

### Conversational UX and Dynamic Overrides
Natural-language requests are first-class inputs.

Examples:
- “Register T1c to T2.”
- “Read DICOM metadata only; what is TE for T2w?”

V2 supports:
- Intent classification and workflow-type inference.
- Modality/path binding from case context.
- Dynamic argument overrides from user phrasing.
- Human-readable terminal responses (separate thought styling + concise final answer).

## Deep DICOM RAG and QA Intent
V2 introduces a specialized metadata QA path:

- `identify_sequences` performs deep DICOM ingestion.
- Per-series headers include:
  - Tag TSV summary.
  - Recursive unrolled dump of nested sequence (`SQ`) elements.
  - Private tags where available.
  - Bulk/pixel data omission for safety and size control.
  - Explicit physical parameter section with TE/TR.
- `rag_search` resolves header index references and answers case-local questions from evidence.

This supports physics/protocol questions like:
- Echo Time (TE)
- Repetition Time (TR)
- Sequence-level parameter inspection from raw DICOM headers

## Security and Safety Model

### CaseScopeGuard (Enterprise-Grade Filesystem Isolation)
Execution is constrained to scoped case/run roots with symlink-aware path validation.

- Blocks unresolved or out-of-scope path references.
- Defends against path traversal and accidental cross-case leakage.
- Allows explicit, controlled external roots when configured.

### Domain and Schema Whitelisting
- Tool availability is domain-scoped (prostate/brain/cardiac).
- Arguments are schema-filtered and validated before invocation.
- Invalid or disallowed tool calls fail fast with structured errors.

Together, these controls provide a safer baseline for clinical-grade automation.

## Interactive Shell (mri_agent_shell)
The shell is now a production-grade operator interface:

- Startup diagnostics (`:doctor`).
- Natural-language planning and execution.
- Template fallback for underspecified requests.
- DAG visualization and stage-aware run traces.
- Cleaner run summaries (debug JSON hidden by default).

Launch:

```bash
cd /home/longz2/common/medgemma/MRI_Agent
pip install -r requirements.txt
pip install -e .
python -m mri_agent_shell
# or
mri-agent
```

Plain REPL mode:

```bash
python -m mri_agent_shell --plain
```

Non-interactive config mode:

```bash
python -m mri_agent_shell --no-prompt
```

## Quickstart (LangGraph Runtime)

### 1) Install dependencies
```bash
pip install -r MRI_Agent/requirements.txt
```

### 2) Download model assets (prostate + BraTS bundles)
```bash
bash MRI_Agent/scripts/setup_models.sh
```

### 3) Optional lesion candidate setup (prostate)
```bash
bash MRI_Agent/scripts/setup_lesion_candidates.sh
# place RRUNet3D weights under MRI_Agent/external/prostate_mri_lesion_seg/weight
```

### 4) Run an end-to-end workflow
Brain example:

```bash
python -m MRI_Agent.agent.langgraph.loop \
  --domain brain \
  --llm-mode server \
  --case-id Brats18_CBICA_AAM_1 \
  --dicom-case-dir /path/to/MRI_Agent/demo/cases/Brats18_CBICA_AAM_1 \
  --max-steps 12 \
  --finalize-with-llm \
  --server-base-url http://127.0.0.1:8000 \
  --server-model Qwen/Qwen3-VL-30B-A3B-Thinking
```

Cardiac example:

```bash
python -m MRI_Agent.agent.langgraph.loop \
  --domain cardiac \
  --llm-mode server \
  --case-id cine_case_001 \
  --dicom-case-dir /path/to/cardiac_cine_case \
  --max-steps 12
```

## LLM/VLM Backends

### OpenAI-compatible local server (vLLM/proxy)
```bash
bash MRI_Agent/scripts/start_vllm_server_apptainer.sh
# or
bash MRI_Agent/scripts/start_medgemma_server_apptainer.sh
```

Then run with:
- `--llm-mode server`
- `--server-base-url`
- `--server-model`

### Cloud APIs
OpenAI:
```bash
export OPENAI_API_KEY=...
python -m MRI_Agent.agent.langgraph.loop --llm-mode openai --api-model gpt-4o-mini ...
```

Anthropic:
```bash
export ANTHROPIC_API_KEY=...
python -m MRI_Agent.agent.langgraph.loop --llm-mode anthropic --api-model claude-3-7-sonnet-20250219 ...
```

Gemini:
```bash
export GOOGLE_API_KEY=...
python -m MRI_Agent.agent.langgraph.loop --llm-mode gemini --api-model gemini-1.5-pro ...
```

## Shell Commands
- `:help`
- `:doctor`
- `:domains`
- `:tools`
- `:paste` (multiline YAML/JSON, end with `:end`)
- `:model`
- `:model set llm=<name> vlm=<name> [provider=<...>] [base_url=<url>] [api_base_url=<url>]`
- `:workspace set <path>`
- `:case load <path>`
- `:run`
- `:exit`

## Artifacts and Traceability
Each run writes auditable outputs under:

```text
MRI_Agent/runs/<case_id>/<run_id>/
```

Typical contents:
- `case_state.json`
- `execution_log.jsonl`
- `artifacts/ingest/` (DICOM inventory, metadata, deep header dumps)
- `artifacts/registration/`
- `artifacts/segmentation/`
- `artifacts/features/`
- `artifacts/vlm/`
- `artifacts/report/`

User-facing exports:

```text
MRI_Agent/reports/<case_id>/report.md
MRI_Agent/logs/<case_id>/tool_trace.jsonl
```

## Experimental and Upcoming

### Experimental Tools
Enable optional tools:

```bash
export MRI_AGENT_ENABLE_EXPERIMENTAL=1
# or individually
export MRI_AGENT_ENABLE_RAG=1
export MRI_AGENT_ENABLE_SANDBOX=1
```

- `rag_search` / `search_repo`: local evidence retrieval.
- `sandbox_exec`: controlled command execution in artifacts workspace.

### Upcoming: Sandbox Code Interpreter
V2 lays the foundation for a richer code-interpreter path:
- Dynamic Python-based analysis tasks.
- Case-scoped execution context.
- Integration with DAG planning and safety gates.

## Configuration and External Assets
Default managed paths:
- `MRI_Agent/models/`
- `MRI_Agent/external/prostate_mri_lesion_seg/`

Environment overrides:
- `MRI_AGENT_MODEL_REGISTRY`
- `MRI_AGENT_PROSTATE_BUNDLE_DIR`
- `MRI_AGENT_BRATS_BUNDLE_DIR`
- `MRI_AGENT_LESION_APP_DIR`
- `MRI_AGENT_LESION_WEIGHTS_DIR`
- `MRI_AGENT_PROSTATE_DISTORTION_ROOT`
- `MRI_AGENT_PROSTATE_DISTORTION_DIFF_CKPT`
- `MRI_AGENT_PROSTATE_DISTORTION_CNN_CKPT`
- `MRI_AGENT_PROSTATE_DISTORTION_TEST_ROOTS`
- `MRI_AGENT_PROSTATE_DISTORTION_PYTHON`

## Benchmark
Simple run scoring:

```bash
python -m MRI_Agent.benchmark.score_run --run-dir MRI_Agent/runs/<case_id>/<run_id>
# or
python -m MRI_Agent.benchmark.score_run --case-id Brats18_CBICA_AAM_1
```

## Repository Docs
- `MRI_Agent/docs/MRI_Agent_framework.md`
- `MRI_Agent/docs/MRI_Agent_tool.md`
- `MRI_Agent/docs/cardiac_acdc_classification_rules.md`

## License and Clinical Notice
This project is a research and engineering framework. It is not a medical device and is not a substitute for clinical judgment.
