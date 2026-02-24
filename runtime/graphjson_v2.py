from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


NodeKind = Literal["io", "tool", "ui_gate", "subgraph"]
NodeStatus = Literal["idle", "running", "success", "error", "skipped", "blocked"]
EdgeKind = Literal["data", "control"]


class NodeArtifactRef(BaseModel):
    artifact_id: str
    kind: str = "unknown"
    path: str = ""
    preview_url: Optional[str] = None


class NodeProvenance(BaseModel):
    planned_by: str = "brain"
    plan_trace_ref: Optional[str] = None


class GraphNodeV2(BaseModel):
    node_id: str
    label: str
    node_kind: NodeKind = "tool"
    tool_candidates: List[str] = Field(default_factory=list)
    tool_selected: str = ""
    tool_locked: Optional[str] = None
    config_schema: Dict[str, Any] = Field(default_factory=dict)
    config_values: Dict[str, Any] = Field(default_factory=dict)
    config_locked_fields: List[str] = Field(default_factory=list)
    status: NodeStatus = "idle"
    artifacts: List[NodeArtifactRef] = Field(default_factory=list)
    provenance: NodeProvenance = Field(default_factory=NodeProvenance)
    meta: Dict[str, Any] = Field(default_factory=dict)
    ports: Dict[str, List[str]] = Field(default_factory=lambda: {"in": ["default"], "out": ["default"]})


class GraphEdgeV2(BaseModel):
    source: str
    source_port: str = "default"
    target: str
    target_port: str = "default"
    edge_kind: EdgeKind = "control"
    condition: Optional[str] = None


class GraphJSONV2(BaseModel):
    graph_id: str
    name: str
    version: str = "v2"
    nodes: List[GraphNodeV2] = Field(default_factory=list)
    edges: List[GraphEdgeV2] = Field(default_factory=list)
    layout: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class NodeOverridePatch(BaseModel):
    tool_locked: Optional[str] = None
    config_values: Dict[str, Any] = Field(default_factory=dict)
    config_locked_fields: List[str] = Field(default_factory=list)
    skip: Optional[bool] = None
    replan: bool = False
    resume_payload: Dict[str, Any] = Field(default_factory=dict)


class RunPatch(BaseModel):
    node_overrides: Dict[str, NodeOverridePatch] = Field(default_factory=dict)
    replan: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


def merge_run_patch(
    current: Dict[str, Dict[str, Any]],
    incoming: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {str(k): dict(v or {}) for k, v in (current or {}).items()}
    for node_id, patch in (incoming or {}).items():
        nid = str(node_id or "").strip()
        if not nid:
            continue
        cur = dict(merged.get(nid) or {})
        if isinstance(patch, NodeOverridePatch):
            pobj = patch.model_dump(mode="python")
        elif isinstance(patch, dict):
            pobj = dict(patch)
        else:
            continue
        if pobj.get("tool_locked") is not None:
            cur["tool_locked"] = str(pobj.get("tool_locked") or "").strip() or None
        if isinstance(pobj.get("config_values"), dict):
            cv = dict(cur.get("config_values") or {})
            cv.update(dict(pobj.get("config_values") or {}))
            cur["config_values"] = cv
        if isinstance(pobj.get("config_locked_fields"), list):
            cur["config_locked_fields"] = [str(x) for x in pobj.get("config_locked_fields") if str(x).strip()]
        if pobj.get("skip") is not None:
            cur["skip"] = bool(pobj.get("skip"))
        if isinstance(pobj.get("resume_payload"), dict) and pobj.get("resume_payload"):
            rp = dict(cur.get("resume_payload") or {})
            rp.update(dict(pobj.get("resume_payload") or {}))
            cur["resume_payload"] = rp
        cur["replan"] = bool(pobj.get("replan", cur.get("replan", False)))
        merged[nid] = cur
    return merged
