<div align="center">

<img src="https://img.shields.io/badge/MedGemma-Nexus-0f4c81?style=for-the-badge&labelColor=0a2f52&logo=google&logoColor=white" alt="MedGemma Nexus" height="36"/>

# 🧠 MedGemma Nexus

### *Autonomous Volumetric Radiology Intelligence — From Raw DICOM to Clinical Report*

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Framework-1C7C54?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![MedGemma](https://img.shields.io/badge/MedGemma-1.5--4b--it-4285F4?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemma/)
[![License](https://img.shields.io/badge/License-Research%20Only-red?style=flat-square)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

<br/>

> **Bridging the execution gap between AI reasoning and real-world clinical imaging.**  
> MedGemma Nexus elevates Google's MedGemma from a static consultant into a fault-tolerant,  
> self-correcting agent that autonomously navigates the full 3D/4D radiological workflow.

<br/>

```
  RAW DICOM/NIfTI  ──►  [ BCER Agent Framework ]  ──►  CLINICAL REPORT
                           Brain · Cerebellum
                          Extremity · Reflector
```

</div>

---

## 🔬 The Problem We're Solving

Medical AI today lives in **"2D flatland"** — excelling at isolated slice Q&A while failing to replicate what radiologists actually do:

| Traditional AI | MedGemma Nexus |
|---|---|
| Answers questions on pre-selected 2D slices | Orchestrates full 3D/4D volumetric pipelines |
| Brittle to coordinate space mismatches | Self-recovering with MedGemma-guided reflection |
| No auditability trail | Provenance-complete ledger for every AI claim |
| Static second reader | Autonomous orchestrator of the entire workflow |

**The root cause:** Long-horizon volumetric tasks accumulate spatial and tool errors that silently invalidate downstream outputs. No existing system has the fault tolerance to handle raw hospital DICOM data reliably — until now.

---

## 🏗️ Architecture: The BCER Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                     BCER Agent Framework                        │
│                                                                 │
│   ┌─────────────────┐         ┌─────────────────────────────┐  │
│   │   🧠  BRAIN     │         │      🪞  REFLECTOR          │  │
│   │  (MedGemma 1.5) │         │      (MedGemma 1.5)         │  │
│   │                 │         │                             │  │
│   │ High-level      │         │ Failure diagnosis engine.   │  │
│   │ symbolic plan   │         │ Guides localized subgraph   │  │
│   │ generation &    │         │ recovery when edge cases    │  │
│   │ clinical intent │         │ break physical execution.   │  │
│   │ interpretation  │         │                             │  │
│   └────────┬────────┘         └──────────────┬──────────────┘  │
│            │  plan                           │ reflect         │
│            ▼                                 ▲                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │            ⚙️  CEREBELLUM + EXTREMITY                   │  │
│   │         Deterministic Execution Engine (DAG)            │  │
│   │                                                         │  │
│   │  Ingest → Register → Segment → Extract → Synthesize    │  │
│   │                                                         │  │
│   │  • Sandboxed tool execution  • Typed I/O contracts      │  │
│   │  • Symbolic artifact binding • Dependency-aware DAG     │  │
│   └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Why decouple?** MedGemma plans brilliantly — but physical volumetric execution is messy. By separating clinical reasoning from deterministic tool dispatch, we let MedGemma think in clinical concepts while the Cerebellum handles coordinate spaces, transforms, and I/O — without burning LLM context on execution minutiae.

---

## ✨ Key Capabilities

- **🫁 Multi-domain MRI support** — Brain (glioma/BraTS), Cardiac (ACDC), Prostate (PI-RADS)  
- **📦 Native volumetric processing** — raw DICOM/NIfTI ingestion, reconstruction, registration, segmentation  
- **🔁 Two-tier fault tolerance** — deterministic repair first, MedGemma reflection fallback  
- **📋 Full clinical auditability** — every measurement traceable to raw data and tool invocations  
- **🖥️ Comfy-style Graph GUI** — interactive visual studio for workflow inspection and reporting  
- **🔌 OpenAI-compatible serving** — drop-in MedGemma server for easy integration  

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
cd MRI_Agent
conda create -n mriagent python=3.12 -y
conda activate mriagent
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### 2. Download Model Assets

```bash
bash scripts/setup_models.sh
bash scripts/setup_lesion_candidates.sh
```

Place MedGemma at `models/medgemma-1.5-4b-it` (or provide your own path).

### 3. Launch the Full Stack

**Terminal 1 — MedGemma inference server:**
```bash
python scripts/medgemma_openai_server.py \
  --model models/medgemma-1.5-4b-it \
  --host 127.0.0.1 --port 8000
```

**Terminal 2 — Graph GUI server:**
```bash
python -m uvicorn MRI_Agent.ui.graph_gui_server:app --host 0.0.0.0 --port 8787
```

**Verify health:**
```bash
curl http://127.0.0.1:8000/health   # MedGemma server
curl http://127.0.0.1:8787/health   # GUI server
```

Open `http://127.0.0.1:8787` in your browser.

---

## 🧪 Demo Cases (Included)

Three reproducible demo subjects are bundled under `demo/cases/Archive/`:

<details>
<summary><strong>🧠 Brain MRI (BraTS glioma)</strong></summary>

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
</details>

<details>
<summary><strong>❤️ Cardiac MRI (ACDC)</strong></summary>

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
</details>

<details>
<summary><strong>🫀 Prostate MRI (PI-RADS)</strong></summary>

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
</details>

---

## 📁 Output Structure

Every run produces a fully auditable artifact tree:

```
runs/<case_id>/<run_id>/
├── case_state.json              # Full agent state snapshot
├── execution_log.jsonl          # Step-by-step provenance ledger
└── artifacts/
    ├── ingest/                  # Raw series identification
    ├── registration/            # Spatial normalization outputs
    ├── segmentation/            # Volumetric segmentation masks
    ├── features/                # Quantitative measurements
    ├── vlm/                     # MedGemma vision outputs
    └── report/
        └── clinical_report.md   # ← Rendered in GUI report panel
```

---

## 🗂️ Repository Structure

```
MRI_Agent/
├── agent/            # LangGraph planning loops & policy logic
├── commands/         # Tool registry, dispatcher, schema plumbing
├── mri_agent_shell/  # Shell runtime and orchestration layer
├── tools/            # Deterministic clinical tool implementations
├── ui/               # Graph GUI backend (FastAPI) + frontend
├── scripts/          # Server entrypoints & setup utilities
├── runtime/          # Execution environment & artifact binding
└── demo/cases/       # Bundled reproducible demo subjects
```

---

## 💡 Compute Notes

- MedGemma inference requires a node with a valid CUDA driver stack (GPU strongly recommended)
- If `/health` on port `8000` hangs, model loading may be blocked or CPU fallback is too slow
- For HPC/remote setups: create an SSH tunnel from your laptop targeting ports `8000` and `8787`

---

## 👥 Team

<table>
<tr>
<td align="center" width="50%">
<strong>Ziyang Long</strong><br/>
Medical Vision-Language Model Researcher<br/>
<em>Lead Architect · Agent Framework Design</em><br/>
Core infrastructure and BCER logical architecture
</td>
<td align="center" width="50%">
<strong>Yifan Gao</strong><br/>
MD Candidate & Clinical Imaging Expert<br/>
<em>Clinical Interface Lead · System Optimization</em><br/>
UI/UX clinical feasibility, debugging & QA
</td>
</tr>
</table>

---

## ⚠️ Safety Disclaimer

> This project is intended **for research and engineering demonstration only.**  
> It is **not** a certified medical device and **must not** be used for clinical diagnosis  
> or treatment decisions. Always consult qualified medical professionals for clinical care.

---

<div align="center">

*Built with ❤️ on Google MedGemma 1.5 · LangGraph · Python 3.12*

</div>
