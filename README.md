# MedGemma Nexus (MRI_Agent Workspace)

This repository is an end-to-end radiology agent workspace built around:
- LangGraph planning (`brain`)
- deterministic tool execution (`cerebellum`)
- MedGemma-compatible OpenAI-style serving
- Comfy-style GUI (`MedGemma Nexus`)

It supports prostate, brain, and cardiac MRI workflows with auditable run artifacts.

## What Is Included

- Core runtime: `agent/`, `commands/`, `mri_agent_shell/`, `tools/`, `runtime/`
- GUI studio: `ui/graph_gui_server.py`, `ui/static/`
- MedGemma server entrypoint: `scripts/medgemma_openai_server.py`
- Three demo subjects (uploaded for reproducible demos):
  - `demo/cases/Archive/Brats18_CBICA_AAM_1` (brain)
  - `demo/cases/Archive/acdc_multiseq_patient061_ed` (cardiac)
  - `demo/cases/Archive/prostate_health` (prostate)

## Environment Setup

```bash
cd MRI_Agent
conda create -n mriagent python=3.12 -y
conda activate mriagent
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Model Assets

### Pipeline model assets

```bash
bash scripts/setup_models.sh
bash scripts/setup_lesion_candidates.sh
```

### MedGemma model

Place MedGemma at:
- `models/medgemma-1.5-4b-it`

Or provide your own local model path via command line when starting the server.

## Run The Full Stack

### Terminal 1: MedGemma server

```bash
cd MRI_Agent
conda activate mriagent
python scripts/medgemma_openai_server.py \
  --model models/medgemma-1.5-4b-it \
  --host 127.0.0.1 \
  --port 8000
```

### Terminal 2: GUI server

```bash
cd MRI_Agent
conda activate mriagent
python -m uvicorn MRI_Agent.ui.graph_gui_server:app --host 0.0.0.0 --port 8787
```

### Health checks

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8787/health
```

### Open GUI

- Local: `http://127.0.0.1:8787`
- Remote/HPC: create SSH tunnel from your laptop.

## CLI Smoke Commands (Using Included Demo Cases)

### Brain

```bash
python -m MRI_Agent.agent.langgraph.loop \
  --domain brain \
  --case-id demo_brain \
  --dicom-case-dir demo/cases/Archive/Brats18_CBICA_AAM_1 \
  --llm-mode server \
  --server-base-url http://127.0.0.1:8000 \
  --server-model google/medgemma-1.5-4b-it \
  --max-steps 12
```

### Cardiac

```bash
python -m MRI_Agent.agent.langgraph.loop \
  --domain cardiac \
  --case-id demo_cardiac \
  --dicom-case-dir demo/cases/Archive/acdc_multiseq_patient061_ed \
  --llm-mode server \
  --server-base-url http://127.0.0.1:8000 \
  --server-model google/medgemma-1.5-4b-it \
  --max-steps 12
```

### Prostate

```bash
python -m MRI_Agent.agent.langgraph.loop \
  --domain prostate \
  --case-id demo_prostate \
  --dicom-case-dir demo/cases/Archive/prostate_health \
  --llm-mode server \
  --server-base-url http://127.0.0.1:8000 \
  --server-model google/medgemma-1.5-4b-it \
  --max-steps 12
```

## Output Locations

Each run writes to:

```text
runs/<case_id>/<run_id>/
```

Typical outputs:
- `case_state.json`
- `execution_log.jsonl`
- `artifacts/ingest/`
- `artifacts/registration/`
- `artifacts/segmentation/`
- `artifacts/features/`
- `artifacts/vlm/`
- `artifacts/report/clinical_report.md`

GUI report panel renders `clinical_report.md` by default.

## Key Notes For Compute Nodes

- MedGemma should run on a compute node with valid CUDA driver stack.
- If `/health` hangs while port `8000` is open, model loading/inference is blocked or CPU-fallback is too slow.
- If GUI shows `Connection refused`, verify:
  1. MedGemma server is running and healthy.
  2. GUI server is running on `8787`.
  3. SSH tunnel targets the correct node and ports.

## Repository Structure

- `agent/`: LangGraph and policy logic
- `commands/`: registry/dispatcher/schema plumbing
- `mri_agent_shell/`: shell runtime and orchestration
- `tools/`: deterministic clinical tools
- `ui/`: interactive graph studio backend + frontend
- `scripts/`: startup and utility scripts
- `demo/cases/Archive/`: bundled demo subjects

## Safety

This project is for research and engineering demos. It is **not** a clinical device and must not be used for clinical diagnosis or treatment decisions.
