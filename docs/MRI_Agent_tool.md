# MRI_Agent Tool Catalog (Current)

This document summarizes the currently registered tools and where they live.
The tool names are stable; module filenames were cleaned up for readability.

## Tool Registry
Tool contracts + dispatcher live in:
- `MRI_Agent/commands/schemas.py`
- `MRI_Agent/commands/registry.py`
- `MRI_Agent/commands/dispatcher.py`

## Registered Tools (Current)
| Tool name | Module | Purpose |
| --- | --- | --- |
| `ingest_dicom_to_nifti` / `identify_sequences` | `MRI_Agent/tools/dicom_ingest.py` | DICOM header inventory + sequence mapping, optional NIfTI conversion |
| `register_to_reference` | `MRI_Agent/tools/registration.py` | SimpleITK resample into reference space |
| `segment_prostate` | `MRI_Agent/tools/prostate_segmentation.py` | MONAI prostate zonal segmentation |
| `brats_mri_segmentation` | `MRI_Agent/tools/brain_tumor_segmentation.py` | MONAI BraTS tumor subregions |
| `segment_cardiac_cine` | `MRI_Agent/tools/cardiac_cine_segmentation.py` | Cardiac cine segmentation via local cmr_reverse nnUNet (Task900_ACDC_Phys) |
| `classify_cardiac_cine_disease` | `MRI_Agent/tools/cardiac_cine_classification.py` | Rule-based ACDC-style cardiac disease classification (NOR/MINF/DCM/HCM/RV) |
| `extract_roi_features` | `MRI_Agent/tools/roi_features.py` | ROI x sequence feature table + overlays + optional PyRadiomics features |
| `detect_lesion_candidates` | `MRI_Agent/tools/prostate_lesion_candidates.py` | Prostate lesion candidates (RRUNet3D) |
| `correct_prostate_distortion` | `MRI_Agent/tools/prostate_distortion_correction.py` | Prostate DWI/ADC distortion correction via external T2+CNN+diffusion pipeline |
| `package_vlm_evidence` | `MRI_Agent/tools/vlm_evidence.py` | Evidence bundle for VLM/reporting |
| `generate_report` | `MRI_Agent/tools/report_generation.py` | Report synthesis (optionally VLM-backed) |

## Experimental Tools (Opt-in)
Enable with:
```bash
export MRI_AGENT_ENABLE_EXPERIMENTAL=1
# or individually:
export MRI_AGENT_ENABLE_RAG=1
export MRI_AGENT_ENABLE_SANDBOX=1
```

| Tool name | Module | Purpose |
| --- | --- | --- |
| `search_repo` | `MRI_Agent/tools/rag_search.py` | Lightweight repo text search (prototype RAG) |
| `sandbox_exec` | `MRI_Agent/tools/sandbox_exec.py` | Run a shell command in a temp artifacts workdir (prototype, not secure) |

## Minimal Dependencies (MVP)
These are the core libs needed for the registered tools:
- `pydicom`, `SimpleITK`, `numpy`, `pillow`
- `torch`, `monai`, `nibabel`
- Optional: `scikit-image`
- Optional (radiomics): `pyradiomics`

Optional (cloud LLMs): `openai`, `anthropic`, `google-generativeai`.

See `MRI_Agent/requirements.txt` for exact pins.
Model assets are downloaded via `MRI_Agent/scripts/setup_models.sh` (bundles) and `MRI_Agent/scripts/setup_lesion_candidates.sh` (lesion candidates).

## Runner Usage (Recommended)
LangGraph runner:
```bash
python -m MRI_Agent.agent.langgraph.loop \
  --domain brain \
  --llm-mode server \
  --autofix-mode coach \
  --case-id Brats18_CBICA_AAM_1 \
  --dicom-case-dir /path/to/MRI_Agent/demo/cases/Brats18_CBICA_AAM_1 \
  --max-steps 12 \
  --max-new-tokens 2048 \
  --finalize-with-llm \
  --server-base-url http://127.0.0.1:8000 \
  --server-model Qwen/Qwen3-VL-30B-A3B-Thinking
```

Cloud API example (OpenAI):
```bash
export OPENAI_API_KEY=...
python -m MRI_Agent.agent.langgraph.loop \
  --domain brain \
  --llm-mode openai \
  --api-model gpt-4o-mini \
  --api-base-url https://api.openai.com/v1 \
  --finalize-with-llm \
  --case-id Brats18_CBICA_AAM_1 \
  --dicom-case-dir /path/to/MRI_Agent/demo/cases/Brats18_CBICA_AAM_1
```

Reactive runner:
```bash
python -m MRI_Agent.agent.loop --llm-mode server --case-id sub-057 --dicom-case-dir /path/to/case
```

Standalone cardiac script:
```bash
python MRI_Agent/scripts/run_cardiac_cine_seg.py \
  --cine-path /path/to/cine.nii.gz \
  --output-dir /path/to/output
```

Standalone cardiac classification script:
```bash
python MRI_Agent/scripts/run_cardiac_cine_classification.py \
  --seg-path /path/to/cardiac_seg.nii.gz \
  --patient-info-path /path/to/Info.cfg \
  --output-dir /path/to/output
```

## Notes on Evidence + Reports
- `package_vlm_evidence` is packaging-only and does not generate narrative text.
- `generate_report` can run in deterministic mode or call a VLM server / cloud API (`llm_mode=server|openai|anthropic|gemini`).
- Report finalization logic lives in `MRI_Agent/runtime/finalize.py`.

## Shell-Supported Task Families (mri_agent_shell v1)
The interactive shell plans only across currently registered core tools. If a request is outside this surface, shell returns `not supported`.
Domains/capabilities are discovered from registered tool metadata at runtime (not hardcoded in template output).

| Domain | Shell request types | Planned core tools |
| --- | --- | --- |
| prostate | `full_pipeline`, `segment`, `register`, `lesion`, `report` | `identify_sequences`, `register_to_reference`, `segment_prostate`, `extract_roi_features`, `detect_lesion_candidates`, `package_vlm_evidence`, `generate_report` |
| brain | `full_pipeline`, `segment`, `report` | `identify_sequences`, `brats_mri_segmentation`, `extract_roi_features`, `package_vlm_evidence`, `generate_report` |
| cardiac | `full_pipeline`, `segment`, `classify`, `report` | `identify_sequences`, `segment_cardiac_cine`, `classify_cardiac_cine_disease`, `extract_roi_features`, `package_vlm_evidence`, `generate_report` |

Dry-run mode adds dummy tools:
- `dummy_load_case`
- `dummy_segment`
- `dummy_generate_report`

Shell command surface:
- `:help`
- `:doctor`
- `:domains`
- `:tools`
- `:paste` (multiline YAML/JSON paste mode, end with `:end`)
- `:model`
- `:model set llm=<name> vlm=<name> [provider=<openai_official|openai_compatible_server|stub|anthropic|gemini>] [base_url=<url>] [api_base_url=<url>]`
- `:workspace set <path>`
- `:case load <path>`
- `:run`
- `:exit`

Shell draft behavior:
- `domain: brain|prostate|cardiac` auto-populates domain demo `case_ref` and `case_id` when case_ref is empty/default.
- `:case load <path>` pins case context and overrides auto demo defaults.
- Natural-language tasks run an `inspect_case` observe step before planning; plan args are then bound to case-relative absolute paths where discoverable.
- `case.input` is a session key placeholder and is not considered runnable unless bound to a real path.
- If required modality preconditions are missing (e.g., no `cine` for cardiac classify), shell rejects the task instead of building an invalid downstream pipeline.
- Runtime enforces per-step `depends_on` contracts and records blocked steps as `status=BLOCKED` in trace JSONL.

## Removed / Not Registered
Experimental tools that are not in the registry (e.g., segmentation QC and TotalSegmentator prostate) have been removed to keep the core surface minimal and deterministic.
