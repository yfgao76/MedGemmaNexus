from __future__ import annotations

import io
import json
import os
import copy
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agent.langgraph.loop import run_langgraph_agent
from ..commands.registry import Tool, ToolRegistry
from ..commands.schemas import tool_config_schema
from ..commands.schemas import ToolSpec
from ..core.domain_config import get_domain_config
from ..core.paths import project_root
from ..core.plan_dag import AgentPlanDAG, PlanNode, legacy_plan_to_dag
from ..llm.adapter_vllm_server import VLLMServerConfig
from ..mri_agent_shell.agent.brain import Brain
from ..mri_agent_shell.runtime.session import ModelConfig, SessionState
from ..mri_agent_shell.tool_registry import build_shell_registry, discover_domain_catalog
from ..runtime.graphjson import build_graphjson_v2, graphjson_to_mermaid
from ..runtime.graphjson_v2 import RunPatch, merge_run_patch

try:
    from ..mri_agent_shell.runtime.cerebellum import Cerebellum
except Exception:  # pragma: no cover - optional dependency path for lightweight GUI-only runs
    Cerebellum = None  # type: ignore[assignment]


SUPPORTED_LLM_MODES = {"server", "stub", "openai", "anthropic", "gemini"}


@dataclass
class RunSession:
    run_id: str
    case_id: str
    engine: str = "shell"
    domain: str = "prostate"
    request_type: str = "full_pipeline"
    case_ref: str = ""
    status: str = "queued"
    run_dir: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    events: List[Dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    done: bool = False
    dag_obj: Optional[Dict[str, Any]] = None
    graph_json: Optional[Dict[str, Any]] = None
    node_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    run_config: Dict[str, Any] = field(default_factory=dict)
    patch_files: List[str] = field(default_factory=list)
    plan_trace_before: Optional[str] = None
    plan_trace_after: Optional[str] = None

    def push(self, event: Dict[str, Any]) -> None:
        with self.lock:
            self.events.append(event)

    def snapshot(self, idx: int) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.events[idx:])


class StartRunRequest(BaseModel):
    case_id: str = Field(default="demo_case")
    dicom_case_dir: Optional[str] = None
    runs_root: Optional[str] = None
    domain: str = Field(default="prostate")
    llm_mode: str = Field(default="server")
    max_steps: int = Field(default=12)
    autofix_mode: str = Field(default="force")
    finalize_with_llm: bool = Field(default=False)
    max_new_tokens: int = Field(default=1024)
    llm_timeout_s: int = Field(default=45)
    server_base_url: str = Field(default="http://127.0.0.1:8000")
    server_model: str = Field(default="google/medgemma-1.5-4b-it")
    api_model: Optional[str] = None
    api_base_url: Optional[str] = None
    request_type: str = Field(default="full_pipeline")
    engine: str = Field(default="shell")
    start_from_node_id: Optional[str] = None
    invalidate_downstream: bool = Field(default=True)
    node_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class PlanRequest(BaseModel):
    prompt: str = Field(default="")
    domain: str = Field(default="prostate")
    request_type: str = Field(default="full_pipeline")
    case_ref: Optional[str] = None
    case_id: Optional[str] = None


class RerunRequest(BaseModel):
    start_from_node_id: Optional[str] = None
    invalidate_downstream: bool = Field(default=True)


ROOT = project_root()
RUNS: Dict[str, RunSession] = {}
_BRAIN_LOCK = threading.Lock()
_BRAIN_CACHE: Optional[Brain] = None
_REGISTRY_CACHE: Any = None
_PLAN_REGISTRY_CACHE: Any = None

app = FastAPI(title="MedGemma Nexus", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "ui" / "static")), name="static")


def _normalize_llm_mode(raw: str) -> str:
    mode = str(raw or "").strip().lower()
    if mode == "medgemma":
        # Backward compatibility: map prior GUI default onto supported runtime mode.
        return "server"
    if mode in SUPPORTED_LLM_MODES:
        return mode
    return "server"


def _normalize_server_base_url(base_url: str) -> str:
    url = str(base_url or "").strip()
    if not url:
        return "http://127.0.0.1:8000"
    url = url.rstrip("/")
    low = url.lower()
    for suffix in ("/v1/chat/completions", "/chat/completions", "/v1/completions", "/v1"):
        if low.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url.rstrip("/") or "http://127.0.0.1:8000"


def _probe_openai_server(base_url: str, *, timeout_s: float = 3.0) -> Dict[str, Any]:
    root = _normalize_server_base_url(base_url)
    urls = [root + "/health", root + "/v1/models"]
    out: Dict[str, Any] = {
        "ok": False,
        "base_url": root,
        "checked_at": datetime.utcnow().isoformat() + "Z",
        "checks": [],
    }
    for u in urls:
        rec: Dict[str, Any] = {"url": u, "ok": False}
        try:
            with urllib.request.urlopen(u, timeout=float(timeout_s)) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                rec["status"] = int(getattr(resp, "status", 200) or 200)
                rec["ok"] = (200 <= rec["status"] < 300)
                try:
                    rec["json"] = json.loads(raw) if raw else None
                except Exception:
                    rec["text"] = raw[:300]
        except urllib.error.HTTPError as e:
            rec["status"] = int(getattr(e, "code", 0) or 0)
            try:
                body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            except Exception:
                body = ""
            rec["error"] = f"HTTPError({e.code} {getattr(e, 'reason', '')})"
            if body:
                rec["text"] = body[:300]
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        out["checks"].append(rec)
    out["ok"] = any(bool(c.get("ok")) for c in out["checks"])
    return out


def _default_case_ref() -> str:
    env_case = str(os.environ.get("MRI_AGENT_GUI_CASE_REF", "")).strip()
    if env_case:
        p = Path(env_case).expanduser().resolve()
        if p.exists() and p.is_dir():
            return str(p)
    for cand in (
        ROOT / "demo" / "cases" / "sub-057",
        ROOT / "data",
        ROOT / "runs",
        ROOT,
    ):
        if cand.exists() and cand.is_dir():
            return str(cand.resolve())
    return str(ROOT.resolve())


def _goal_for_request(domain: str, request_type: str) -> str:
    dom = str(domain or "prostate").strip().lower() or "prostate"
    req = str(request_type or "full_pipeline").strip().lower() or "full_pipeline"
    return f"Run {dom} MRI {req} workflow and generate report-ready outputs."


def _get_shell_brain() -> Brain:
    global _BRAIN_CACHE, _PLAN_REGISTRY_CACHE
    if _BRAIN_CACHE is not None:
        return _BRAIN_CACHE
    with _BRAIN_LOCK:
        if _BRAIN_CACHE is None:
            if _PLAN_REGISTRY_CACHE is None:
                _PLAN_REGISTRY_CACHE = _build_planning_registry()
            reg = _PLAN_REGISTRY_CACHE
            domain_catalog = discover_domain_catalog(reg)
            _BRAIN_CACHE = Brain(tool_specs=reg.list_specs(), domain_catalog=domain_catalog)
    return _BRAIN_CACHE


def _build_planning_registry() -> ToolRegistry:
    reg = ToolRegistry()

    def _noop(args: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
        del args, ctx
        return {"data": {}}

    specs = [
        ToolSpec(
            name="identify_sequences",
            description="Identify MRI sequences in a case folder.",
            input_schema={
                "type": "object",
                "properties": {
                    "dicom_case_dir": {"type": "string"},
                    "convert_to_nifti": {"type": "boolean"},
                    "output_subdir": {"type": "string"},
                },
                "required": ["dicom_case_dir"],
            },
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["ingest", "identify", "dicom"],
        ),
        ToolSpec(
            name="register_to_reference",
            description="Register moving sequence to fixed reference.",
            input_schema={
                "type": "object",
                "properties": {
                    "fixed": {"type": "string"},
                    "moving": {"type": "string"},
                    "output_subdir": {"type": "string"},
                },
                "required": ["fixed", "moving"],
            },
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["registration", "register", "alignment"],
        ),
        ToolSpec(
            name="alignment_qc",
            description="Registration quality control.",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["registration", "qc", "alignment"],
        ),
        ToolSpec(
            name="materialize_registration",
            description="Materialize transforms/output previews.",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["registration", "materialize", "qc"],
        ),
        ToolSpec(
            name="segment_prostate",
            description="Prostate segmentation.",
            input_schema={
                "type": "object",
                "properties": {
                    "t2w_ref": {"type": "string"},
                    "output_subdir": {"type": "string"},
                },
                "required": ["t2w_ref"],
            },
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["segmentation", "prostate", "segment"],
        ),
        ToolSpec(
            name="brats_mri_segmentation",
            description="Brain tumor segmentation.",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["segmentation", "brain", "segment"],
        ),
        ToolSpec(
            name="segment_cardiac_cine",
            description="Cardiac cine segmentation.",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["segmentation", "cardiac", "segment"],
        ),
        ToolSpec(
            name="classify_cardiac_cine_disease",
            description="Cardiac disease classification.",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["classification", "cardiac", "report"],
        ),
        ToolSpec(
            name="extract_roi_features",
            description="Extract ROI features.",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["features", "extract"],
        ),
        ToolSpec(
            name="detect_lesion_candidates",
            description="Lesion candidate extraction.",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["lesion", "extract", "prostate"],
        ),
        ToolSpec(
            name="correct_prostate_distortion",
            description="Distortion correction.",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["prostate", "distortion", "registration"],
        ),
        ToolSpec(
            name="package_vlm_evidence",
            description="Package visual evidence for report.",
            input_schema={
                "type": "object",
                "properties": {
                    "case_state_path": {"type": "string"},
                    "output_subdir": {"type": "string"},
                },
                "required": ["case_state_path"],
            },
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["report", "evidence", "vlm"],
        ),
        ToolSpec(
            name="generate_report",
            description="Generate final report.",
            input_schema={
                "type": "object",
                "properties": {
                    "case_state_path": {"type": "string"},
                    "output_subdir": {"type": "string"},
                    "domain": {"type": "string"},
                },
                "required": ["case_state_path", "domain"],
            },
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["report", "vlm", "final"],
        ),
        ToolSpec(
            name="rag_search",
            description="Repository/context search.",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["rag", "qa"],
        ),
        ToolSpec(
            name="sandbox_exec",
            description="Sandbox command execution.",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object", "properties": {}, "required": []},
            version="planning",
            tags=["sandbox", "custom_analysis"],
        ),
    ]
    for spec in specs:
        reg.register(Tool(spec=spec, func=_noop))
    return reg


def _get_planning_registry():
    global _PLAN_REGISTRY_CACHE
    if _PLAN_REGISTRY_CACHE is not None:
        return _PLAN_REGISTRY_CACHE
    with _BRAIN_LOCK:
        if _PLAN_REGISTRY_CACHE is None:
            _PLAN_REGISTRY_CACHE = _build_planning_registry()
    return _PLAN_REGISTRY_CACHE


def _get_shell_registry():
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    with _BRAIN_LOCK:
        if _REGISTRY_CACHE is None:
            _REGISTRY_CACHE = build_shell_registry(dry_run=False, include_core=True)
    return _REGISTRY_CACHE


def _build_gui_dag(
    *,
    domain: str,
    request_type: str,
    case_ref: str,
    case_id: Optional[str],
) -> Dict[str, Any]:
    case_path = Path(str(case_ref or _default_case_ref())).expanduser().resolve()
    case_id_val = str(case_id or case_path.name or "demo_case").strip() or "demo_case"

    session = SessionState(
        workspace_path=str(ROOT.resolve()),
        runs_root=str((ROOT / "runs").resolve()),
        model_config=ModelConfig(provider="stub", llm="stub"),
        dry_run=False,
    )
    if case_path.exists() and case_path.is_dir():
        session.set_case_input(str(case_path))
    session.set_case_id(case_id_val)

    case_ref_arg = "case.input" if session.case_inputs.get("case_input_path") else str(case_path)
    tpl = {
        "REQUIRED": {
            "domain": str(domain),
            "request_type": str(request_type),
            "case_ref": case_ref_arg,
        },
        "OPTIONAL": {
            "case_id": case_id_val,
        },
    }

    brain = _get_shell_brain()
    res = brain.from_template(tpl, session=session, goal=_goal_for_request(domain, request_type))
    if res.kind != "plan" or res.plan is None:
        raise RuntimeError(res.message or "unable to build DAG plan from shell brain")

    legacy = res.plan.model_dump()
    dag = legacy_plan_to_dag(
        legacy_plan=legacy,
        domain=str(domain or "prostate").strip().lower() or "prostate",
        case_id=case_id_val,
        case_ref=(str(session.case_inputs.get("case_input_path") or case_path)),
        workspace_root=str(ROOT.resolve()),
        runs_root=str((ROOT / "runs").resolve()),
        plan_id=f"gui_{case_id_val}",
    )
    return dict(dag.model_dump())


def _infer_node_kind(node_type: str, tool_name: str) -> str:
    nt = str(node_type or "tool").strip().lower()
    tn = str(tool_name or "").strip().lower()
    if nt in {"gate", "reflect"}:
        return "ui_gate"
    if tn in {"identify_sequences", "ingest_dicom_to_nifti"}:
        return "io"
    return "tool"


def _dag_to_graphjson(
    dag_obj: Dict[str, Any],
    *,
    domain: str,
    request_type: str,
    case_ref: str,
    node_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    raw_nodes = dag_obj.get("nodes") if isinstance(dag_obj.get("nodes"), list) else []
    by_node_id: Dict[str, Dict[str, Any]] = {}
    for node in raw_nodes:
        if isinstance(node, dict) and str(node.get("node_id") or "").strip():
            by_node_id[str(node.get("node_id"))] = dict(node)

    reg = _get_planning_registry()
    override_map = {str(k): dict(v or {}) for k, v in (node_overrides or {}).items()}

    nodes: List[Dict[str, Any]] = [
        {
            "node_id": "START",
            "label": "START",
            "node_kind": "io",
            "tool_candidates": [],
            "tool_selected": "",
            "status": "idle",
            "config_schema": {"type": "object", "properties": {}, "required": []},
            "config_values": {},
            "provenance": {"planned_by": "brain", "plan_trace_ref": None},
            "meta": {"stage": "entry"},
            "ports": {"in": ["default"], "out": ["default"]},
        },
        {
            "node_id": "END",
            "label": "END",
            "node_kind": "io",
            "tool_candidates": [],
            "tool_selected": "",
            "status": "idle",
            "config_schema": {"type": "object", "properties": {}, "required": []},
            "config_values": {},
            "provenance": {"planned_by": "brain", "plan_trace_ref": None},
            "meta": {"stage": "exit"},
            "ports": {"in": ["default"], "out": ["default"]},
        },
    ]

    edges_set: Set[Tuple[str, str, str]] = set()
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or "").strip()
        if not node_id:
            continue
        tool_name = str(node.get("tool_name") or "")
        selected = str(node.get("tool_selected") or tool_name)
        candidates = [str(x) for x in (node.get("tool_candidates") or []) if str(x).strip()]
        if selected and selected not in candidates:
            candidates.insert(0, selected)

        override = override_map.get(node_id) or {}
        locked = str(override.get("tool_locked") or node.get("tool_locked") or "").strip() or None
        if locked and locked not in candidates:
            candidates.insert(0, locked)

        try:
            spec = reg.get_spec(selected or tool_name)
            fallback_schema = tool_config_schema(spec)
        except Exception:
            fallback_schema = {"type": "object", "properties": {}, "required": []}

        config_schema = dict(node.get("config_schema") or fallback_schema)
        config_values = dict(node.get("arguments") or {})
        if isinstance(node.get("config_values"), dict):
            config_values.update(dict(node.get("config_values") or {}))
        if isinstance(override.get("config_values"), dict):
            config_values.update(dict(override.get("config_values") or {}))

        status = str(node.get("status") or "idle").strip().lower() or "idle"
        if status not in {"idle", "running", "success", "error", "skipped", "blocked"}:
            status = "idle"

        nodes.append(
            {
                "node_id": node_id,
                "label": str(node.get("label") or tool_name or node_id),
                "node_kind": _infer_node_kind(str(node.get("node_type") or "tool"), selected or tool_name),
                "tool_candidates": sorted(set(candidates), key=lambda x: (0 if x == selected else 1, x)),
                "tool_selected": selected or tool_name,
                "tool_locked": locked,
                "config_schema": config_schema,
                "config_values": config_values,
                "config_locked_fields": [str(x) for x in (node.get("config_locked_fields") or []) if str(x).strip()],
                "status": status,
                "artifacts": [dict(x) for x in (node.get("artifacts") or []) if isinstance(x, dict)],
                "provenance": dict(node.get("provenance") or {"planned_by": "brain", "plan_trace_ref": None}),
                "meta": {
                    "stage": str(node.get("stage") or "misc"),
                    "node_type": str(node.get("node_type") or "tool"),
                    "required": bool(node.get("required", True)),
                    "selection_rationale": str(node.get("selection_rationale") or ""),
                    "skip": bool(override.get("skip", node.get("skip", False))),
                },
                "ports": {"in": ["default"], "out": ["default"]},
            }
        )

    node_id_set = {str(n.get("node_id")) for n in nodes}
    prev_declared_node_id = ""
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        nid = str(node.get("node_id") or "").strip()
        if not nid:
            continue
        deps = [str(x).strip() for x in (node.get("depends_on") or []) if str(x).strip()]
        if not deps:
            # Keep dependency-free nodes in declared plan order so the visual DAG
            # tracks the actual shell executor path.
            if prev_declared_node_id and prev_declared_node_id in node_id_set:
                edges_set.add((prev_declared_node_id, nid, "seq"))
            else:
                edges_set.add(("START", nid, ""))
            prev_declared_node_id = nid
            continue
        for dep in deps:
            if dep in node_id_set:
                edges_set.add((dep, nid, ""))
            else:
                edges_set.add(("START", nid, f"missing:{dep}"))
        prev_declared_node_id = nid

    outgoing = {src for src, _, _ in edges_set}
    for nid in list(node_id_set):
        if nid in {"START", "END"}:
            continue
        if nid not in outgoing:
            edges_set.add((nid, "END", ""))

    edge_records: List[Dict[str, Any]] = []
    for src, tgt, cond in sorted(edges_set):
        if src not in node_id_set or tgt not in node_id_set:
            continue
        rec: Dict[str, Any] = {
            "source": src,
            "source_port": "default",
            "target": tgt,
            "target_port": "default",
            "edge_kind": "control",
        }
        if cond:
            rec["condition"] = cond
        edge_records.append(rec)

    return build_graphjson_v2(
        graph_id=f"mri-agent-dag-{domain}-{request_type}",
        name=f"MRI Agent DAG ({domain}/{request_type})",
        version="v2",
        nodes=nodes,
        edges=edge_records,
    )


def _fallback_graphjson(domain: str, request_type: str, reason: str) -> Dict[str, Any]:
    return build_graphjson_v2(
        graph_id=f"mri-agent-fallback-{domain}-{request_type}",
        name=f"MRI Agent Fallback Graph ({domain}/{request_type})",
        version="v2",
        nodes=[
            {"node_id": "START", "label": "START", "node_kind": "io"},
            {"node_id": "identify_sequences_000", "label": "identify_sequences", "node_kind": "tool"},
            {"node_id": "generate_report_001", "label": "generate_report", "node_kind": "tool"},
            {
                "node_id": "END",
                "label": "END",
                "node_kind": "io",
                "meta": {"reason": reason},
            },
        ],
        edges=[
            {"source": "START", "target": "identify_sequences_000", "edge_kind": "control"},
            {"source": "identify_sequences_000", "target": "generate_report_001", "edge_kind": "control"},
            {"source": "generate_report_001", "target": "END", "edge_kind": "control"},
        ],
    )


def _graphjson(
    *,
    domain: str = "prostate",
    request_type: str = "full_pipeline",
    case_ref: Optional[str] = None,
    case_id: Optional[str] = None,
    node_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    dom = str(domain or "prostate").strip().lower() or "prostate"
    req = str(request_type or "full_pipeline").strip().lower() or "full_pipeline"
    cref = str(case_ref or _default_case_ref())
    try:
        dag_obj = _build_gui_dag(domain=dom, request_type=req, case_ref=cref, case_id=case_id)
        return _dag_to_graphjson(
            dag_obj,
            domain=dom,
            request_type=req,
            case_ref=cref,
            node_overrides=node_overrides,
        )
    except Exception as e:
        return _fallback_graphjson(dom, req, reason=f"{type(e).__name__}: {e}")


def _list_case_run_dirs(runs_root: Path, case_id: str) -> List[Path]:
    case_root = runs_root / case_id
    if not case_root.exists() or not case_root.is_dir():
        return []
    dirs = [p for p in case_root.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: p.stat().st_mtime)


def _discover_new_run_dir(
    runs_root: Path,
    case_id: str,
    known_names: Set[str],
    *,
    created_after_epoch: float,
) -> Optional[Path]:
    candidates = _list_case_run_dirs(runs_root, case_id)
    fresh: List[Path] = []
    for p in candidates:
        if p.name in known_names:
            continue
        try:
            if p.stat().st_mtime + 0.5 < created_after_epoch:
                continue
        except OSError:
            continue
        fresh.append(p)
    if fresh:
        return fresh[-1]
    return None


def _read_jsonl_delta(path: Path, start_line: int) -> Tuple[List[Dict[str, Any]], int]:
    if not path.exists():
        return [], start_line
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return [], start_line
    if start_line >= len(lines):
        return [], len(lines)
    recs: List[Dict[str, Any]] = []
    for ln in lines[start_line:]:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict):
            recs.append(obj)
    return recs, len(lines)


def _artifact_relpath(path_like: Any, run_dir: Path) -> str:
    raw = str(path_like or "").strip()
    if not raw:
        return ""
    try:
        p = Path(raw).expanduser().resolve()
        rr = run_dir.resolve()
        if p.is_absolute():
            rel = p.relative_to(rr)
            return rel.as_posix()
    except Exception:
        pass
    s = raw.replace("\\", "/")
    if s.startswith("artifacts/"):
        return s
    if "/artifacts/" in s:
        return s.split("/artifacts/", 1)[-1] if s.endswith("/artifacts/") else "artifacts/" + s.split("/artifacts/", 1)[-1]
    return s


def _normalize_source_event(evt: Dict[str, Any], run_dir: Path, *, gui_run_id: str, case_id: str) -> Dict[str, Any]:
    out = dict(evt or {})
    out.setdefault("run_id", gui_run_id)
    out.setdefault("case_id", case_id)
    if not out.get("node_id") and out.get("tool_name"):
        out["node_id"] = str(out.get("tool_name"))
    if out.get("event_type") == "artifact" and isinstance(out.get("artifact"), dict):
        art = dict(out.get("artifact") or {})
        art["path"] = _artifact_relpath(art.get("path"), run_dir)
        out["artifact"] = art
    return out


def _emit_execution_entry(
    session: RunSession,
    rec: Dict[str, Any],
    run_dir: Path,
    *,
    case_id: str,
    step_counter: List[int],
) -> None:
    tool_name = str(rec.get("tool_name") or "unknown_tool")
    node_id = tool_name
    step = int(step_counter[0])
    ts = str(rec.get("timestamp") or (datetime.utcnow().isoformat() + "Z"))

    session.push(
        {
            "event_type": "node_start",
            "node_id": node_id,
            "step": step,
            "tool_name": tool_name,
            "ts": ts,
            "run_id": session.run_id,
            "case_id": case_id,
        }
    )

    session.push(
        {
            "event_type": "tool_call",
            "node_id": node_id,
            "step": step,
            "tool_name": tool_name,
            "stage": rec.get("stage"),
            "arguments": rec.get("arguments") if isinstance(rec.get("arguments"), dict) else {},
            "ts": ts,
            "run_id": session.run_id,
            "case_id": case_id,
        }
    )

    ok = bool(rec.get("ok"))
    session.push(
        {
            "event_type": "tool_result",
            "node_id": node_id,
            "step": step,
            "tool_name": tool_name,
            "ok": ok,
            "warnings": rec.get("warnings") if isinstance(rec.get("warnings"), list) else [],
            "error": rec.get("error") if isinstance(rec.get("error"), dict) else None,
            "runtime_ms": rec.get("runtime_ms"),
            "ts": ts,
            "run_id": session.run_id,
            "case_id": case_id,
        }
    )

    generated = rec.get("generated_artifacts")
    if isinstance(generated, list):
        for art in generated:
            if not isinstance(art, dict):
                continue
            art_out = dict(art)
            art_out["path"] = _artifact_relpath(art_out.get("path"), run_dir)
            session.push(
                {
                    "event_type": "artifact",
                    "node_id": node_id,
                    "step": step,
                    "tool_name": tool_name,
                    "artifact": art_out,
                    "ts": ts,
                    "run_id": session.run_id,
                    "case_id": case_id,
                }
            )

    session.push(
        {
            "event_type": "node_end",
            "node_id": node_id,
            "step": step,
            "tool_name": tool_name,
            "status": ("ok" if ok else "fail"),
            "ts": ts,
            "run_id": session.run_id,
            "case_id": case_id,
        }
    )
    step_counter[0] = step + 1


def _node_id_from_call_id(call_id: Any) -> str:
    s = str(call_id or "").strip()
    if not s.startswith("dag_") or "_try" not in s:
        return ""
    try:
        body = s[len("dag_") :]
        return body.rsplit("_try", 1)[0]
    except Exception:
        return ""


def _session_model_config_from_request(req: StartRunRequest) -> ModelConfig:
    mode = _normalize_llm_mode(req.llm_mode)
    if mode == "stub":
        return ModelConfig(provider="stub", llm="stub", vlm="stub", max_tokens=int(req.max_new_tokens), temperature=0.0)
    if mode == "openai":
        return ModelConfig(
            provider="openai_official",
            llm=str(req.api_model or "gpt-4o-mini"),
            api_base_url=str(req.api_base_url or ""),
            max_tokens=int(req.max_new_tokens),
            temperature=0.0,
        )
    if mode == "anthropic":
        return ModelConfig(
            provider="anthropic",
            llm=str(req.api_model or "claude-3-7-sonnet-20250219"),
            max_tokens=int(req.max_new_tokens),
            temperature=0.0,
        )
    if mode == "gemini":
        return ModelConfig(
            provider="gemini",
            llm=str(req.api_model or "gemini-1.5-pro"),
            max_tokens=int(req.max_new_tokens),
            temperature=0.0,
        )
    return ModelConfig(
        provider="openai_compatible_server",
        llm=str(req.server_model or "google/medgemma-1.5-4b-it"),
        server_base_url=_normalize_server_base_url(req.server_base_url).rstrip("/") + "/v1",
        max_tokens=int(req.max_new_tokens),
        temperature=0.0,
    )


def _push_dispatcher_event(
    session: RunSession,
    evt: Dict[str, Any],
    *,
    case_id: str,
    runs_root: Optional[Path] = None,
) -> None:
    kind = str(evt.get("event_type") or "").strip().lower()
    runner_run_id = str(evt.get("run_id") or "").strip()
    if runner_run_id and not session.run_dir:
        try:
            rr = Path(runs_root or (ROOT / "runs")).expanduser().resolve()
            guessed = rr / case_id / runner_run_id
            session.run_dir = str(guessed)
            session.push(
                {
                    "event_type": "run_bound",
                    "run_id": session.run_id,
                    "case_id": case_id,
                    "runner_run_id": runner_run_id,
                    "run_dir": session.run_dir,
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )
        except Exception:
            pass
    node_id = _node_id_from_call_id(evt.get("call_id")) or str(evt.get("node_id") or evt.get("tool_name") or "")
    base = {
        "run_id": session.run_id,
        "case_id": case_id,
        "node_id": node_id,
        "tool_name": str(evt.get("tool_name") or ""),
        "call_id": str(evt.get("call_id") or ""),
        "ts": str(evt.get("ts") or datetime.utcnow().isoformat() + "Z"),
    }
    if kind == "tool_call":
        session.push(
            {
                **base,
                "event_type": "node_start",
                "status": "RUNNING",
                "arguments": dict(evt.get("arguments") or {}),
            }
        )
        session.push(
            {
                **base,
                "event_type": "tool_call",
                "stage": str(evt.get("stage") or ""),
                "arguments": dict(evt.get("arguments") or {}),
            }
        )
        return
    if kind == "tool_result":
        ok = bool(evt.get("ok"))
        session.push(
            {
                **base,
                "event_type": "tool_result",
                "ok": ok,
                "warnings": list(evt.get("warnings") or []),
                "runtime_ms": evt.get("runtime_ms"),
                "error": evt.get("error"),
            }
        )
        session.push(
            {
                **base,
                "event_type": "node_end",
                "status": ("ok" if ok else "fail"),
            }
        )
        return
    if kind == "artifact":
        art = dict(evt.get("artifact") or {})
        if session.run_dir and art.get("path"):
            try:
                ap = Path(str(art.get("path"))).expanduser().resolve()
                rr = Path(session.run_dir).expanduser().resolve()
                art["path"] = ap.relative_to(rr).as_posix()
            except Exception:
                pass
        if art:
            session.push({**base, "event_type": "artifact", "artifact": art})
        return
    session.push({**base, "event_type": (kind or "log"), "payload": dict(evt)})


def _start_shell_worker(session: RunSession, req: StartRunRequest) -> None:
    session.status = "running"
    session.done = False
    case_id = str(req.case_id or "demo_case")
    runs_root = Path(req.runs_root).expanduser().resolve() if req.runs_root else (ROOT / "runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    if Cerebellum is None:
        session.status = "failed"
        session.error = (
            "Shell executor is unavailable. Missing runtime dependency 'jsonschema' "
            "or related shell runtime package."
        )
        session.push(
            {
                "event_type": "run_error",
                "run_id": session.run_id,
                "case_id": case_id,
                "error": session.error,
                "ts": datetime.utcnow().isoformat() + "Z",
            }
        )
        session.done = True
        return

    case_ref = str(req.dicom_case_dir or session.case_ref or _default_case_ref())
    session.case_ref = case_ref
    session.domain = str(req.domain or session.domain or "prostate")
    session.request_type = str(req.request_type or session.request_type or "full_pipeline")
    session.node_overrides = merge_run_patch(session.node_overrides, req.node_overrides)

    try:
        dag_obj = _build_gui_dag(
            domain=session.domain,
            request_type=session.request_type,
            case_ref=case_ref,
            case_id=case_id,
        )
        timeout_override = max(5, int(req.llm_timeout_s or 45))
        if timeout_override > 0:
            for n in list(dag_obj.get("nodes") or []):
                if not isinstance(n, dict):
                    continue
                if str(n.get("tool_name") or "").strip() != "generate_report":
                    continue
                node_id = str(n.get("node_id") or "").strip()
                if not node_id:
                    continue
                ov = dict(session.node_overrides.get(node_id) or {})
                cv = dict(ov.get("config_values") or {})
                if "llm_timeout_s" not in cv:
                    cv["llm_timeout_s"] = int(timeout_override)
                ov["config_values"] = cv
                session.node_overrides[node_id] = ov
        session.dag_obj = copy.deepcopy(dag_obj)
        session.graph_json = _dag_to_graphjson(
            dag_obj,
            domain=session.domain,
            request_type=session.request_type,
            case_ref=case_ref,
            node_overrides=session.node_overrides,
        )
    except Exception as e:
        session.status = "failed"
        session.error = f"failed to build plan DAG: {type(e).__name__}: {e}"
        session.push(
            {
                "event_type": "run_error",
                "run_id": session.run_id,
                "case_id": case_id,
                "error": session.error,
                "ts": datetime.utcnow().isoformat() + "Z",
            }
        )
        session.done = True
        return

    model_cfg = _session_model_config_from_request(req)
    shell_session = SessionState(
        workspace_path=str(ROOT.resolve()),
        runs_root=str(runs_root),
        model_config=model_cfg,
        dry_run=False,
    )
    try:
        cref = Path(case_ref).expanduser().resolve()
        if cref.exists() and cref.is_dir():
            shell_session.set_case_input(str(cref))
    except Exception:
        pass
    shell_session.set_case_id(case_id)

    session.push(
        {
            "event_type": "run_start",
            "run_id": session.run_id,
            "case_id": case_id,
            "engine": "shell",
            "domain": session.domain,
            "request_type": session.request_type,
            "case_ref": case_ref,
            "llm_timeout_s": max(5, int(req.llm_timeout_s or 45)),
            "node_overrides": session.node_overrides,
            "ts": datetime.utcnow().isoformat() + "Z",
        }
    )

    cere = Cerebellum(
        session=shell_session,
        registry=_get_shell_registry(),
        max_attempts=2,
    )
    try:
        out = cere.execute_dag(
            AgentPlanDAG.model_validate(session.dag_obj),
            emit=lambda msg: session.push(
                {
                    "event_type": "log",
                    "run_id": session.run_id,
                    "case_id": case_id,
                    "message": str(msg),
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            ),
            start_from_node_id=(str(req.start_from_node_id).strip() if req.start_from_node_id else None),
            invalidate_downstream=bool(req.invalidate_downstream),
            node_overrides=session.node_overrides,
            event_sink=lambda evt: _push_dispatcher_event(
                session,
                evt,
                case_id=case_id,
                runs_root=runs_root,
            ),
        )
        session.run_dir = str(out.get("run_dir") or "")
        if session.run_dir:
            session.push(
                {
                    "event_type": "run_bound",
                    "run_id": session.run_id,
                    "case_id": case_id,
                    "runner_run_id": Path(session.run_dir).name,
                    "run_dir": session.run_dir,
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )
        session.status = "completed"
        session.push(
            {
                "event_type": "run_end",
                "run_id": session.run_id,
                "case_id": case_id,
                "status": "ok",
                "run_dir": session.run_dir,
                "ts": datetime.utcnow().isoformat() + "Z",
            }
        )
    except Exception as e:
        session.status = "failed"
        session.error = repr(e)
        session.push(
            {
                "event_type": "run_error",
                "run_id": session.run_id,
                "case_id": case_id,
                "error": session.error,
                "ts": datetime.utcnow().isoformat() + "Z",
            }
        )
    finally:
        session.done = True


def _session_run_dir(session: RunSession) -> Optional[Path]:
    if not session.run_dir:
        return None
    try:
        p = Path(session.run_dir).expanduser().resolve()
    except Exception:
        return None
    if not p.exists() or not p.is_dir():
        return None
    return p


def _persist_patch_file(session: RunSession, patch_obj: Dict[str, Any]) -> str:
    base = _session_run_dir(session)
    if base is None:
        base = ROOT / "runs" / "_gui_patch_cache" / session.run_id
    patch_dir = base / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    path = patch_dir / f"patch_{ts}.json"
    path.write_text(json.dumps(patch_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(path)


def _resolve_report_path(run_dir: Path) -> Optional[Path]:
    for rel in (
        Path("artifacts/report/clinical_report.md"),
        Path("artifacts/report/report.md"),
        Path("artifacts/final/final_report.md"),
    ):
        p = run_dir / rel
        if p.exists() and p.is_file():
            return p
    clinicals = sorted(run_dir.glob("artifacts/**/clinical_report.md"))
    if clinicals:
        return clinicals[0]
    cands = sorted(run_dir.glob("artifacts/**/report.md"))
    if cands:
        return cands[0]
    finals = sorted(run_dir.glob("artifacts/**/final_report.md"))
    if finals:
        return finals[0]
    return None


def _collect_evidence(run_dir: Path, run_id: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _push(path_like: Any, *, caption: str = "", provenance: Optional[Dict[str, Any]] = None) -> None:
        raw = str(path_like or "").strip()
        if not raw:
            return
        p = (run_dir / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        try:
            rel = p.relative_to(run_dir).as_posix()
        except Exception:
            return
        if rel in seen:
            return
        seen.add(rel)
        items.append(
            {
                "artifact_id": rel.replace("/", "_"),
                "path": rel,
                "caption": caption or Path(rel).name,
                "url": f"/runs/{run_id}/artifact?path={urllib.parse.quote(rel, safe='')}",
                "provenance": dict(provenance or {}),
            }
        )

    idx_candidates = [
        run_dir / "artifacts" / "evidence" / "index.json",
        run_dir / "artifacts" / "vlm" / "evidence" / "index.json",
        run_dir / "artifacts" / "evidence" / "manifest.json",
    ]
    for idx in idx_candidates:
        if not idx.exists() or not idx.is_file():
            continue
        try:
            obj = json.loads(idx.read_text(encoding="utf-8"))
        except Exception:
            continue
        recs: List[Any] = []
        if isinstance(obj, list):
            recs = obj
        elif isinstance(obj, dict):
            recs = list(obj.get("items") or obj.get("evidence") or [])
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            prov = {
                "node_id": rec.get("node_id"),
                "tool_name": rec.get("tool_name"),
                "tool_call_id": rec.get("tool_call_id"),
            }
            imgs = rec.get("images")
            if isinstance(imgs, list) and imgs:
                for img in imgs:
                    if isinstance(img, dict):
                        _push(
                            img.get("path"),
                            caption=str(img.get("caption") or rec.get("caption") or ""),
                            provenance=prov,
                        )
                continue
            _push(rec.get("path"), caption=str(rec.get("caption") or ""), provenance=prov)

    for p in sorted(run_dir.glob("artifacts/**/*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            continue
        _push(str(p), caption=p.name, provenance={})
    return items


@app.get("/")
def home() -> FileResponse:
    return FileResponse(str(ROOT / "ui" / "static" / "index.html"))


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


@app.get("/server/probe")
def probe_server(
    base_url: str = Query(default="http://127.0.0.1:8000"),
    timeout_s: float = Query(default=3.0, ge=0.2, le=20.0),
) -> Dict[str, Any]:
    return _probe_openai_server(base_url=base_url, timeout_s=float(timeout_s))


@app.get("/graph")
def get_graph(
    domain: str = Query(default="prostate"),
    request_type: str = Query(default="full_pipeline"),
    case_ref: Optional[str] = Query(default=None),
    case_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return _graphjson(domain=domain, request_type=request_type, case_ref=case_ref, case_id=case_id)


@app.post("/cases/{case_id}/plan")
def plan_case(case_id: str, req: PlanRequest) -> Dict[str, Any]:
    dom = str(req.domain or "prostate").strip().lower() or "prostate"
    req_type = str(req.request_type or "full_pipeline").strip().lower() or "full_pipeline"
    cref = str(req.case_ref or _default_case_ref())
    cid = str(case_id or req.case_id or "demo_case")
    dag_obj = _build_gui_dag(domain=dom, request_type=req_type, case_ref=cref, case_id=cid)
    graph = _dag_to_graphjson(
        dag_obj,
        domain=dom,
        request_type=req_type,
        case_ref=cref,
        node_overrides=None,
    )
    return {
        "case_id": cid,
        "domain": dom,
        "request_type": req_type,
        "case_ref": cref,
        "graph": graph,
        "dag": dag_obj,
        "prompt": str(req.prompt or ""),
    }


@app.get("/graph/mermaid")
def get_graph_mermaid(
    domain: str = Query(default="prostate"),
    request_type: str = Query(default="full_pipeline"),
    case_ref: Optional[str] = Query(default=None),
    case_id: Optional[str] = Query(default=None),
) -> PlainTextResponse:
    graph = _graphjson(domain=domain, request_type=request_type, case_ref=case_ref, case_id=case_id)
    return PlainTextResponse(graphjson_to_mermaid(graph), media_type="text/plain")


@app.get("/graph/png")
def get_graph_png():
    graph = _graphjson()
    return JSONResponse(
        {
            "status": "unavailable",
            "note": "PNG export is not enabled in this build. Use /graph or /graph/mermaid.",
            "graph_id": graph.get("graph_id"),
        },
        status_code=501,
    )


def _start_worker(session: RunSession, req: StartRunRequest) -> None:
    runs_root = Path(req.runs_root).expanduser().resolve() if req.runs_root else (ROOT / "runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    case_id = str(req.case_id or "demo_case")
    session.status = "running"

    requested_mode = str(req.llm_mode or "").strip().lower()
    llm_mode = _normalize_llm_mode(requested_mode)
    if requested_mode == "medgemma":
        session.push(
            {
                "event_type": "warning",
                "run_id": session.run_id,
                "case_id": case_id,
                "message": "llm_mode=medgemma is not a direct run_langgraph_agent mode; using llm_mode=server.",
            }
        )
    elif requested_mode and requested_mode not in SUPPORTED_LLM_MODES:
        session.push(
            {
                "event_type": "warning",
                "run_id": session.run_id,
                "case_id": case_id,
                "message": f"Unsupported llm_mode='{requested_mode}'. Falling back to llm_mode=server.",
            }
        )

    if llm_mode == "server":
        probe = _probe_openai_server(base_url=req.server_base_url, timeout_s=2.5)
        session.push(
            {
                "event_type": "server_probe",
                "run_id": session.run_id,
                "case_id": case_id,
                "probe": probe,
                "ts": datetime.utcnow().isoformat() + "Z",
            }
        )
        if not bool(probe.get("ok")):
            root = str(probe.get("base_url") or req.server_base_url or "http://127.0.0.1:8000")
            session.push(
                {
                    "event_type": "warning",
                    "run_id": session.run_id,
                    "case_id": case_id,
                    "message": (
                        f"Cannot reach llm_mode=server endpoint at '{root}'. "
                        "If MedGemma runs on a compute node, set server_base_url to "
                        "http://<compute-node>:8000 or create a tunnel: "
                        "ssh -N -L 8000:127.0.0.1:8000 <compute-node>."
                    ),
                }
            )

    known_runs = {p.name for p in _list_case_run_dirs(runs_root, case_id)}
    source_mode = ""
    offsets: Dict[str, int] = {"events": 0, "execution": 0}
    step_counter = [0]
    discovered_run_dir: Optional[Path] = None
    runner_out: Dict[str, Any] = {}
    runner_err: Dict[str, Any] = {}

    session.push(
        {
            "event_type": "run_start",
            "run_id": session.run_id,
            "case_id": case_id,
            "inputs": {
                "domain": req.domain,
                "dicom_case_dir": req.dicom_case_dir,
                "llm_mode": llm_mode,
                "server_base_url": (req.server_base_url if llm_mode == "server" else None),
                "server_model": (req.server_model if llm_mode == "server" else None),
            },
            "ts": datetime.utcnow().isoformat() + "Z",
        }
    )

    def _runner() -> None:
        try:
            runner_out["run_dir"] = run_langgraph_agent(
                case_id=case_id,
                dicom_case_dir=req.dicom_case_dir,
                runs_root=runs_root,
                llm_mode=llm_mode,
                autofix_mode=req.autofix_mode,
                max_steps=int(req.max_steps),
                finalize_with_llm=bool(req.finalize_with_llm),
                max_new_tokens=int(req.max_new_tokens),
                server_cfg=VLLMServerConfig(
                    base_url=req.server_base_url,
                    model=req.server_model,
                    max_tokens=int(req.max_new_tokens),
                ),
                api_model=req.api_model,
                api_base_url=req.api_base_url,
                domain=get_domain_config(req.domain),
            )
        except Exception as e:
            runner_err["error"] = e

    worker = threading.Thread(target=_runner, daemon=True)
    runner_started_epoch = time.time()
    worker.start()

    while worker.is_alive():
        if discovered_run_dir is None:
            discovered_run_dir = _discover_new_run_dir(
                runs_root,
                case_id,
                known_runs,
                created_after_epoch=runner_started_epoch,
            )
            if discovered_run_dir is not None:
                session.run_dir = str(discovered_run_dir)
                session.push(
                    {
                        "event_type": "run_bound",
                        "run_id": session.run_id,
                        "case_id": case_id,
                        "runner_run_id": discovered_run_dir.name,
                        "run_dir": str(discovered_run_dir),
                        "ts": datetime.utcnow().isoformat() + "Z",
                    }
                )

        if discovered_run_dir is not None:
            events_path = discovered_run_dir / "events.jsonl"
            exec_path = discovered_run_dir / "execution_log.jsonl"
            if not source_mode:
                if events_path.exists():
                    source_mode = "events"
                elif exec_path.exists():
                    source_mode = "execution"

            if source_mode == "events":
                recs, offsets["events"] = _read_jsonl_delta(events_path, offsets["events"])
                for rec in recs:
                    session.push(
                        _normalize_source_event(
                            rec,
                            discovered_run_dir,
                            gui_run_id=session.run_id,
                            case_id=case_id,
                        )
                    )
            elif source_mode == "execution":
                recs, offsets["execution"] = _read_jsonl_delta(exec_path, offsets["execution"])
                for rec in recs:
                    _emit_execution_entry(
                        session,
                        rec,
                        discovered_run_dir,
                        case_id=case_id,
                        step_counter=step_counter,
                    )

        time.sleep(0.25)

    worker.join()

    final_run_dir: Optional[Path] = None
    if isinstance(runner_out.get("run_dir"), Path):
        final_run_dir = Path(runner_out["run_dir"]).resolve()
    elif isinstance(runner_out.get("run_dir"), str):
        final_run_dir = Path(str(runner_out["run_dir"])).expanduser().resolve()
    elif discovered_run_dir is not None:
        final_run_dir = discovered_run_dir.resolve()

    if final_run_dir is not None:
        session.run_dir = str(final_run_dir)
        if discovered_run_dir is None:
            session.push(
                {
                    "event_type": "run_bound",
                    "run_id": session.run_id,
                    "case_id": case_id,
                    "runner_run_id": final_run_dir.name,
                    "run_dir": str(final_run_dir),
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )
        events_path = final_run_dir / "events.jsonl"
        exec_path = final_run_dir / "execution_log.jsonl"
        if not source_mode:
            if events_path.exists():
                source_mode = "events"
            elif exec_path.exists():
                source_mode = "execution"
        if source_mode == "events":
            recs, offsets["events"] = _read_jsonl_delta(events_path, offsets["events"])
            for rec in recs:
                session.push(
                    _normalize_source_event(
                        rec,
                        final_run_dir,
                        gui_run_id=session.run_id,
                        case_id=case_id,
                    )
                )
        elif source_mode == "execution":
            recs, offsets["execution"] = _read_jsonl_delta(exec_path, offsets["execution"])
            for rec in recs:
                _emit_execution_entry(
                    session,
                    rec,
                    final_run_dir,
                    case_id=case_id,
                    step_counter=step_counter,
                )

    if "error" in runner_err:
        session.status = "failed"
        session.error = repr(runner_err["error"])
        session.push(
            {
                "event_type": "run_error",
                "run_id": session.run_id,
                "case_id": case_id,
                "error": session.error,
                "ts": datetime.utcnow().isoformat() + "Z",
            }
        )
    else:
        session.status = "completed"
        session.push(
            {
                "event_type": "run_end",
                "run_id": session.run_id,
                "case_id": case_id,
                "status": "ok",
                "run_dir": session.run_dir,
                "ts": datetime.utcnow().isoformat() + "Z",
            }
        )

    session.done = True


@app.post("/runs")
def start_run(req: StartRunRequest) -> Dict[str, Any]:
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    engine = str(req.engine or "shell").strip().lower() or "shell"
    if engine not in {"shell", "langgraph"}:
        engine = "shell"
    session = RunSession(
        run_id=run_id,
        case_id=req.case_id,
        engine=engine,
        domain=str(req.domain or "prostate"),
        request_type=str(req.request_type or "full_pipeline"),
        case_ref=str(req.dicom_case_dir or _default_case_ref()),
        node_overrides=merge_run_patch({}, req.node_overrides),
        run_config=req.model_dump(mode="python"),
    )
    RUNS[run_id] = session
    target = _start_shell_worker if engine == "shell" else _start_worker
    th = threading.Thread(target=target, args=(session, req), daemon=True)
    th.start()
    return {
        "run_id": run_id,
        "status": session.status,
        "case_id": req.case_id,
        "engine": engine,
    }


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    session = RUNS.get(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "run_id": session.run_id,
        "case_id": session.case_id,
        "engine": session.engine,
        "domain": session.domain,
        "request_type": session.request_type,
        "case_ref": session.case_ref,
        "status": session.status,
        "run_dir": session.run_dir,
        "error": session.error,
        "created_at": session.created_at,
        "event_count": len(session.events),
    }


@app.get("/runs/{run_id}/graph")
def get_run_graph(run_id: str) -> Dict[str, Any]:
    session = RUNS.get(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="run not found")
    if session.graph_json is not None:
        return dict(session.graph_json)
    if session.dag_obj is not None:
        graph = _dag_to_graphjson(
            session.dag_obj,
            domain=session.domain,
            request_type=session.request_type,
            case_ref=session.case_ref or _default_case_ref(),
            node_overrides=session.node_overrides,
        )
        session.graph_json = graph
        return graph
    return _graphjson(
        domain=session.domain,
        request_type=session.request_type,
        case_ref=session.case_ref or _default_case_ref(),
        case_id=session.case_id,
        node_overrides=session.node_overrides,
    )


@app.post("/runs/{run_id}/patch")
def patch_run(run_id: str, patch: RunPatch) -> Dict[str, Any]:
    session = RUNS.get(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="run not found")

    patch_obj = patch.model_dump(mode="python", exclude_unset=True)
    session.node_overrides = merge_run_patch(session.node_overrides, patch_obj.get("node_overrides") or {})
    patch_path = _persist_patch_file(session, patch_obj)
    session.patch_files.append(patch_path)

    if session.dag_obj is None:
        session.dag_obj = _build_gui_dag(
            domain=session.domain,
            request_type=session.request_type,
            case_ref=session.case_ref or _default_case_ref(),
            case_id=session.case_id,
        )

    needs_replan = bool(patch.replan) or any(
        bool(v.get("replan"))
        for v in (patch_obj.get("node_overrides") or {}).values()
        if isinstance(v, dict)
    )
    if needs_replan:
        before = copy.deepcopy(session.dag_obj)
        session.plan_trace_before = _persist_patch_file(
            session,
            {"event": "plan_trace_before", "dag": before},
        )
        session.dag_obj = _build_gui_dag(
            domain=session.domain,
            request_type=session.request_type,
            case_ref=session.case_ref or _default_case_ref(),
            case_id=session.case_id,
        )
        session.plan_trace_after = _persist_patch_file(
            session,
            {"event": "plan_trace_after", "dag": session.dag_obj},
        )

    session.graph_json = _dag_to_graphjson(
        session.dag_obj,
        domain=session.domain,
        request_type=session.request_type,
        case_ref=session.case_ref or _default_case_ref(),
        node_overrides=session.node_overrides,
    )
    return {
        "run_id": run_id,
        "node_overrides": session.node_overrides,
        "graph": session.graph_json,
        "patch_path": patch_path,
        "replanned": needs_replan,
        "plan_trace_before": session.plan_trace_before,
        "plan_trace_after": session.plan_trace_after,
    }


@app.post("/runs/{run_id}/rerun")
def rerun_run(run_id: str, req: RerunRequest) -> Dict[str, Any]:
    session = RUNS.get(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="run not found")
    if session.status == "running" and not session.done:
        raise HTTPException(status_code=409, detail="run already in progress")

    cfg = dict(session.run_config or {})
    cfg["case_id"] = session.case_id
    cfg["domain"] = session.domain
    cfg["request_type"] = session.request_type
    cfg["dicom_case_dir"] = session.case_ref
    cfg["engine"] = session.engine or "shell"
    cfg["node_overrides"] = session.node_overrides
    cfg["start_from_node_id"] = req.start_from_node_id
    cfg["invalidate_downstream"] = bool(req.invalidate_downstream)
    run_req = StartRunRequest.model_validate(cfg)

    session.status = "queued"
    session.error = None
    session.done = False
    session.push(
        {
            "event_type": "rerun_request",
            "run_id": session.run_id,
            "case_id": session.case_id,
            "start_from_node_id": req.start_from_node_id,
            "invalidate_downstream": bool(req.invalidate_downstream),
            "ts": datetime.utcnow().isoformat() + "Z",
        }
    )
    target = _start_shell_worker if (session.engine == "shell") else _start_worker
    th = threading.Thread(target=target, args=(session, run_req), daemon=True)
    th.start()
    return {
        "run_id": run_id,
        "status": session.status,
        "start_from_node_id": req.start_from_node_id,
        "invalidate_downstream": bool(req.invalidate_downstream),
    }


def _sse_stream(session: RunSession, replay: bool, speed: float):
    idx = 0
    sent = set()
    while True:
        events = session.snapshot(idx)
        if events:
            for evt in events:
                key = json.dumps(evt, sort_keys=True, ensure_ascii=False)
                if replay and key in sent:
                    continue
                payload = json.dumps(evt, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                if replay:
                    sent.add(key)
                    time.sleep(max(0.01, 0.2 / max(speed, 0.1)))
            idx += len(events)
        elif session.done:
            break
        else:
            time.sleep(0.2)
    yield f"data: {json.dumps({'event_type': 'stream_end', 'run_id': session.run_id})}\n\n"


@app.get("/runs/{run_id}/events")
def run_events(
    run_id: str,
    replay: bool = Query(default=False),
    speed: float = Query(default=1.0, ge=0.1, le=10.0),
):
    session = RUNS.get(run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="run not found")
    return StreamingResponse(_sse_stream(session, replay=replay, speed=float(speed)), media_type="text/event-stream")


@app.get("/runs/{run_id}/artifacts")
def list_artifacts(run_id: str) -> Dict[str, Any]:
    session = RUNS.get(run_id)
    if session is None or not session.run_dir:
        raise HTTPException(status_code=404, detail="run or artifacts not found")
    run_dir = Path(session.run_dir)
    items: List[Dict[str, Any]] = []
    for p in sorted(run_dir.glob("artifacts/**/*")):
        if not p.is_file():
            continue
        rel = p.relative_to(run_dir).as_posix()
        items.append({"path": rel, "size": p.stat().st_size})
    return {"run_id": run_id, "artifacts": items}


@app.get("/runs/{run_id}/artifact")
def get_artifact(run_id: str, path: str = Query(...)) -> FileResponse:
    session = RUNS.get(run_id)
    if session is None or not session.run_dir:
        raise HTTPException(status_code=404, detail="run not found")
    base = Path(session.run_dir).resolve()
    tgt = (base / path).resolve()
    if not str(tgt).startswith(str(base)):
        raise HTTPException(status_code=400, detail="invalid artifact path")
    if not tgt.exists() or not tgt.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(str(tgt))


@app.get("/runs/{run_id}/report")
def get_report(run_id: str) -> Dict[str, Any]:
    session = RUNS.get(run_id)
    if session is None or not session.run_dir:
        raise HTTPException(status_code=404, detail="run not found")
    run_dir = Path(session.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="run directory not found")

    report_path = _resolve_report_path(run_dir)
    markdown = ""
    rel = ""
    if report_path is not None and report_path.exists():
        try:
            markdown = report_path.read_text(encoding="utf-8")
            rel = report_path.relative_to(run_dir).as_posix()
        except Exception:
            markdown = ""
            rel = ""
    evidence = _collect_evidence(run_dir, run_id)
    return {
        "run_id": run_id,
        "report_path": rel,
        "markdown": markdown,
        "evidence": evidence,
    }


@app.get("/runs/{run_id}/bundle")
def bundle_run(run_id: str):
    session = RUNS.get(run_id)
    if session is None or not session.run_dir:
        raise HTTPException(status_code=404, detail="run not found")
    run_dir = Path(session.run_dir)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="run directory not found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(run_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(run_dir))
    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{run_id}_bundle.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)
