from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from .graphjson_v2 import GraphEdgeV2, GraphJSONV2, GraphNodeV2, NodeArtifactRef, NodeProvenance


def _to_node_record(node_id: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(node_id),
        "label": str(raw.get("label") or node_id),
        "kind": str(raw.get("kind") or "agent"),
        "meta": dict(raw.get("meta") or {}),
        "ports": dict(raw.get("ports") or {"in": ["default"], "out": ["default"]}),
    }


def _topo_levels(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    *,
    order_index: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    indeg: Dict[str, int] = defaultdict(int)
    adj: Dict[str, List[str]] = defaultdict(list)
    node_ids = {str(n["id"]) for n in nodes}
    for nid in node_ids:
        indeg[nid] = 0
    for e in edges:
        s = str(e.get("source") or "")
        t = str(e.get("target") or "")
        if s not in node_ids or t not in node_ids:
            continue
        adj[s].append(t)
        indeg[t] += 1

    def _sort_key(nid: str) -> Tuple[int, str]:
        if isinstance(order_index, dict):
            return (int(order_index.get(nid, 10**9)), nid)
        return (10**9, nid)

    if isinstance(order_index, dict):
        roots = [nid for nid in sorted(node_ids, key=_sort_key) if indeg[nid] == 0]
    else:
        roots = [nid for nid in sorted(node_ids) if indeg[nid] == 0]
    q = deque(roots)
    level: Dict[str, int] = {nid: 0 for nid in q}
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            level[nxt] = max(level.get(nxt, 0), level.get(cur, 0) + 1)
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    for nid in node_ids:
        level.setdefault(nid, 0)
    return level


def build_graphjson(
    *,
    graph_id: str,
    name: str,
    version: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> Dict[str, Any]:
    node_records = [_to_node_record(str(n.get("id")), n) for n in nodes]
    edge_records: List[Dict[str, Any]] = []
    for e in edges:
        edge_records.append(
            {
                "source": str(e.get("source")),
                "source_port": str(e.get("source_port") or "default"),
                "target": str(e.get("target")),
                "target_port": str(e.get("target_port") or "default"),
                **({"condition": str(e.get("condition"))} if e.get("condition") is not None else {}),
            }
        )

    order_index = {str(n["id"]): i for i, n in enumerate(node_records)}
    levels = _topo_levels(node_records, edge_records, order_index=order_index)
    grouped: Dict[int, List[str]] = defaultdict(list)
    for nid, lv in levels.items():
        grouped[int(lv)].append(nid)
    layout: Dict[str, Dict[str, int]] = {}
    for lv in sorted(grouped.keys()):
        col = sorted(grouped[lv], key=lambda nid: (order_index.get(str(nid), 10**9), str(nid)))
        for row, nid in enumerate(col):
            layout[nid] = {"x": 260 * int(lv), "y": 140 * int(row)}

    return {
        "graph_id": graph_id,
        "name": name,
        "version": version,
        "nodes": node_records,
        "edges": edge_records,
        "layout": layout,
    }


def build_graphjson_v2(
    *,
    graph_id: str,
    name: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    version: str = "v2",
) -> Dict[str, Any]:
    node_records: List[GraphNodeV2] = []
    edge_records: List[GraphEdgeV2] = []

    for raw in nodes:
        node_id = str(raw.get("node_id") or raw.get("id") or "").strip()
        if not node_id:
            continue
        node_records.append(
            GraphNodeV2(
                node_id=node_id,
                label=str(raw.get("label") or node_id),
                node_kind=str(raw.get("node_kind") or raw.get("kind") or "tool"),
                tool_candidates=[str(x) for x in (raw.get("tool_candidates") or []) if str(x).strip()],
                tool_selected=str(raw.get("tool_selected") or raw.get("tool_name") or ""),
                tool_locked=(str(raw.get("tool_locked")).strip() if raw.get("tool_locked") is not None else None),
                config_schema=dict(raw.get("config_schema") or {}),
                config_values=dict(raw.get("config_values") or {}),
                config_locked_fields=[str(x) for x in (raw.get("config_locked_fields") or []) if str(x).strip()],
                status=str(raw.get("status") or "idle"),
                artifacts=[
                    NodeArtifactRef(
                        artifact_id=str(a.get("artifact_id") or a.get("path") or ""),
                        kind=str(a.get("kind") or "unknown"),
                        path=str(a.get("path") or ""),
                        preview_url=(str(a.get("preview_url")) if a.get("preview_url") else None),
                    )
                    for a in (raw.get("artifacts") or [])
                    if isinstance(a, dict)
                ],
                provenance=NodeProvenance(
                    planned_by=str((raw.get("provenance") or {}).get("planned_by") or "brain"),
                    plan_trace_ref=(
                        str((raw.get("provenance") or {}).get("plan_trace_ref"))
                        if (raw.get("provenance") or {}).get("plan_trace_ref")
                        else None
                    ),
                ),
                meta=dict(raw.get("meta") or {}),
                ports=dict(raw.get("ports") or {"in": ["default"], "out": ["default"]}),
            )
        )

    for raw in edges:
        src = str(raw.get("source") or "").strip()
        tgt = str(raw.get("target") or "").strip()
        if not src or not tgt:
            continue
        edge_records.append(
            GraphEdgeV2(
                source=src,
                source_port=str(raw.get("source_port") or "default"),
                target=tgt,
                target_port=str(raw.get("target_port") or "default"),
                edge_kind=str(raw.get("edge_kind") or "control"),
                condition=(str(raw.get("condition")) if raw.get("condition") is not None else None),
            )
        )

    order_index = {n.node_id: i for i, n in enumerate(node_records)}
    levels = _topo_levels(
        [{"id": n.node_id} for n in node_records],
        [{"source": e.source, "target": e.target} for e in edge_records],
        order_index=order_index,
    )
    grouped: Dict[int, List[str]] = defaultdict(list)
    for nid, lv in levels.items():
        grouped[int(lv)].append(nid)
    layout: Dict[str, Dict[str, int]] = {}
    for lv in sorted(grouped.keys()):
        col = sorted(grouped[lv], key=lambda nid: (order_index.get(str(nid), 10**9), str(nid)))
        for row, nid in enumerate(col):
            layout[nid] = {"x": 280 * int(lv), "y": 170 * int(row)}

    graph = GraphJSONV2(
        graph_id=str(graph_id),
        name=str(name),
        version=str(version or "v2"),
        nodes=node_records,
        edges=edge_records,
        layout=layout,
    )
    return graph.model_dump(mode="python")


def graphjson_to_mermaid(graph: Dict[str, Any]) -> str:
    lines: List[str] = ["flowchart LR"]
    for n in graph.get("nodes", []) or []:
        nid = str(n.get("id") or n.get("node_id") or "")
        if not nid:
            continue
        label = str(n.get("label") or nid).replace('"', "'")
        lines.append(f'  {nid}["{label}"]')
    for e in graph.get("edges", []) or []:
        src = str(e.get("source"))
        tgt = str(e.get("target"))
        cond = str(e.get("condition") or "").strip()
        if cond:
            lines.append(f"  {src} -->|{cond}| {tgt}")
        else:
            lines.append(f"  {src} --> {tgt}")
    return "\n".join(lines) + "\n"


def infer_graph_from_compiled(compiled: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    graph_obj = None
    try:
        if hasattr(compiled, "get_graph"):
            graph_obj = compiled.get_graph()
    except Exception:
        graph_obj = None
    if graph_obj is None:
        return [], []

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    try:
        raw_nodes = getattr(graph_obj, "nodes", None)
        if isinstance(raw_nodes, dict):
            for key, val in raw_nodes.items():
                nid = str(key)
                label = None
                if isinstance(val, dict):
                    label = val.get("name") or val.get("label")
                else:
                    label = getattr(val, "name", None) or getattr(val, "label", None)
                nodes.append({"id": nid, "label": str(label or nid), "kind": "agent", "ports": {"in": ["default"], "out": ["default"]}})
        elif isinstance(raw_nodes, list):
            for val in raw_nodes:
                nid = str(getattr(val, "id", None) or getattr(val, "name", None) or "")
                if not nid:
                    continue
                label = str(getattr(val, "name", None) or nid)
                nodes.append({"id": nid, "label": label, "kind": "agent", "ports": {"in": ["default"], "out": ["default"]}})
    except Exception:
        nodes = []

    try:
        raw_edges = getattr(graph_obj, "edges", None)
        if isinstance(raw_edges, list):
            for ee in raw_edges:
                src = getattr(ee, "source", None)
                tgt = getattr(ee, "target", None)
                if src is None and isinstance(ee, (tuple, list)) and len(ee) >= 2:
                    src, tgt = ee[0], ee[1]
                if src is None or tgt is None:
                    continue
                edges.append({"source": str(src), "source_port": "default", "target": str(tgt), "target_port": "default"})
    except Exception:
        edges = []

    return nodes, edges
