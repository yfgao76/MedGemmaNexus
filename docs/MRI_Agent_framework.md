# MRI_Agent Framework (Current Architecture)

## Purpose
MRI_Agent is a tool-oriented MRI agent framework. A VLM/LLM acts as the coordinator while deterministic tools perform imaging work. The system is designed for reproducibility, traceability, and modular replacement of tools.

Key principles:
- The orchestrator is not the imaging algorithm.
- Every tool has strict JSON I/O + artifacts + provenance.
- Case-level state is persisted and recoverable.
- Tool outputs are structured for downstream reporting.

## Current Entrypoints
- Reactive loop: `python -m MRI_Agent.agent.loop`
- LangGraph runner (recommended): `python -m MRI_Agent.agent.langgraph.loop`

## LLM Backends (Current)
- **server**: OpenAI-compatible vLLM server (local or remote).
- **openai / anthropic / gemini**: direct cloud API calls (requires API keys).
- **stub**: deterministic FakeLLM for debugging.

## High-Level Flow (Both Runners)
1) Ingest + identify sequences (DICOM headers, optional NIfTI conversion)
2) Register moving sequences to a reference (T2w for prostate, T1c for brain)
3) Segment ROI (prostate gland or brain tumor subregions)
4) Extract ROI features
5) Package VLM evidence (images + summaries)
6) Generate report (optionally via VLM)

The tool names are stable across runs. The filesystem artifacts are recorded in `case_state.json` and `execution_log.jsonl` under `MRI_Agent/runs/<case_id>/<run_id>/`.

## Core Concepts
- **ToolSpec**: name, description, input/output schema, version, tags.
- **ToolCall**: tool name + JSON args + call id + case id + stage.
- **ToolResult**: ok/error + data + artifacts + provenance.
- **CaseState**: persisted run state; all tool outputs + artifact index.

## Experimental Extensions (Opt-in)
MRI_Agent includes lightweight prototypes that are disabled by default:
- **RAG (local)**: `search_repo` tool for quick text search over the repo.
- **Sandbox (prototype)**: `sandbox_exec` tool to run shell commands in a temp artifacts workdir.

Enable them via environment variables:
```bash
export MRI_AGENT_ENABLE_EXPERIMENTAL=1
# or:
export MRI_AGENT_ENABLE_RAG=1
export MRI_AGENT_ENABLE_SANDBOX=1
```
These are convenience utilities, not hardened security features.

## Repository Layout (Current)
- `MRI_Agent/agent/` orchestration logic (reactive + LangGraph)
  - `agent/loop.py` reactive runner
  - `agent/langgraph/loop.py` LangGraph cyclic runner
  - `agent/subagents/` planner / reflector / policy
  - `agent/hooks/` preconditions + circuit breaker
  - `agent/rules/` tool validation rules
  - `agent/skills/` skill registry (from `config/skills.json`)
- `MRI_Agent/llm/` LLM adapters (vLLM server + cloud APIs)
- `MRI_Agent/commands/` tool registry + dispatcher + schemas
- `MRI_Agent/tools/` tool implementations (see tool catalog)
- `MRI_Agent/runtime/` memory + tool manifest + report finalization
- `MRI_Agent/config/skills.json` skill definitions
- `MRI_Agent/demo/` sample cases (kept as-is)
- `MRI_Agent/example_report/` example outputs (kept as-is)

## Domain Behavior
- **Prostate**
  - Reference: T2w
  - Registration: ADC/DWI -> T2w
  - Segmentation: prostate zones + whole gland
  - Optional lesion candidates

- **Brain (BraTS)**
  - Reference: T1c (fallback T1)
  - Registration: T2/FLAIR -> T1c
  - Segmentation: tumor subregions (TC/WT/ET)

## Output Artifacts (Typical)
- `case_state.json`, `execution_log.jsonl`
- `artifacts/ingest/*` series inventory + DICOM metadata
- `artifacts/registration/*` resampled NIfTI + transforms
- `artifacts/segmentation/*` ROI masks
- `artifacts/features/*` features.csv + slice summary + overlays
- `artifacts/vlm/*` evidence bundle
- `artifacts/report/*` report.json / report.txt

## Notes
- External model assets (MONAI bundles, lesion weights) are expected under `MRI_Agent/models` and `MRI_Agent/external` by default.
- Environment variables override paths: `MRI_AGENT_MODEL_REGISTRY`, `MRI_AGENT_PROSTATE_BUNDLE_DIR`, `MRI_AGENT_BRATS_BUNDLE_DIR`, `MRI_AGENT_LESION_APP_DIR`, `MRI_AGENT_LESION_WEIGHTS_DIR`.
- See `MRI_Agent/README.md` for setup scripts and runtime examples.

## Interactive Shell Architecture (mri_agent_shell)
The shell adds an interactive operator-facing layer without changing core tool algorithms.

Startup/config behavior:
- Priority: `CLI flags > config file > env vars > defaults > optional prompt`.
- Prompt supports `ENTER` for defaults and `?` for per-field help.
- Startup runs a self-check (`:doctor`) once.
- Domain-switch UX: draft context auto-binds demo case paths by domain (unless user pinned case via `:case load`) and re-derives `case_id` on domain changes.

Main modules:
- `mri_agent_shell/agent/brain.py`
  - converts natural-language requests into either:
    - template request (`REQUIRED` / `OPTIONAL`)
    - structured execution plan (tool sequence)
    - unsupported message
  - NL-first operation uses Observe -> Think -> Plan:
    - observe: `inspect_case` scans case content/modality hints
    - think: infer/refine `domain` + `request_type`
    - plan: compile deterministic tool sequence
- `mri_agent_shell/runtime/session.py`
  - tracks global shell state (workspace, case, model config, path keys, current plan)
- `mri_agent_shell/runtime/cerebellum.py`
  - deterministic executor for plan steps
  - path-key resolution
  - tool-level path isolation: normalizes modality tokens to case-relative absolute paths before dispatch
  - arg repair (`tools.arg_models.repair_tool_args`)
  - step dependency contract checks (`depends_on`) with explicit `BLOCKED` status when upstream steps fail/skip
  - schema validation + retries
  - structured event stream emission
- `mri_agent_shell/shell/repl.py`
  - command parser + interactive loop (`:help`, `:doctor`, `:domains`, `:tools`, `:paste`, `:model`, `:workspace`, `:case`, `:run`, `:exit`)
  - multiline paste mode (`:paste` ... `:end`) + persistent template draft context
- `mri_agent_shell/shell/tui.py`
  - optional Textual pane UI; automatic fallback to plain REPL if Textual is unavailable
- `mri_agent_shell/io/report_writer.py`
  - exports final report into workspace-level report path

Event stream (`tool_trace.jsonl`) fields:
- `event_type`
- `timestamp`
- `tool_name`
- `args_summary`
- `status`
- `duration_s`
- `outputs`
- `error`
- `attempt`
- `case_id`
- `run_id`

Shell provider names:
- `openai_official` (OpenAI API key from env/config)
- `openai_compatible_server` (vLLM/proxy base URL)
- `stub` (offline test mode)
- aliases: `openai`, `server`
