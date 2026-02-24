let cy = null;
let evtSource = null;
let lastRunId = null;
let currentGraph = null;
let selectedNodeId = "";
let localNodeOverrides = {};
let eventLines = [];
let artifactPaths = [];

const eventsEl = document.getElementById("events");
const eventFilterEl = document.getElementById("eventFilter");
const artifactsEl = document.getElementById("artifacts");
const artifactFilterEl = document.getElementById("artifactFilter");
const evidenceEl = document.getElementById("evidenceGallery");
const runStatusEl = document.getElementById("runStatus");
const runStatusPillEl = document.getElementById("runStatusPill");
const serverStatusEl = document.getElementById("serverStatus");
const serverStatusPillEl = document.getElementById("serverStatusPill");
const graphStatsEl = document.getElementById("graphStats");
const graphStatsPillEl = document.getElementById("graphStatsPill");
const renderReportBtnEl = document.getElementById("renderReportBtn");
const reportMetaEl = document.getElementById("reportMeta");
const reportViewerEl = document.getElementById("reportViewer");
const openReportRawLinkEl = document.getElementById("openReportRawLink");
const inspectorNodeNameEl = document.getElementById("inspectorNodeName");
const nodeRationaleEl = document.getElementById("nodeRationale");
const toolCandidatesEl = document.getElementById("toolCandidates");
const agentSelectedToolEl = document.getElementById("agentSelectedTool");
const lockToolToggleEl = document.getElementById("lockToolToggle");
const skipNodeToggleEl = document.getElementById("skipNodeToggle");
const configEditorEl = document.getElementById("configEditor");

let reportPath = "";

function setPillState(pillEl, state) {
  if (!pillEl) return;
  pillEl.dataset.state = String(state || "idle");
}

function setRunStatus(text, state = "idle") {
  if (runStatusEl) runStatusEl.textContent = text || "";
  setPillState(runStatusPillEl, state);
}

function setServerStatus(text, state = "idle") {
  if (serverStatusEl) serverStatusEl.textContent = text || "";
  setPillState(serverStatusPillEl, state);
}

function updateGraphStats() {
  if (!graphStatsEl) return;
  const nodes = Array.isArray(currentGraph && currentGraph.nodes) ? currentGraph.nodes : [];
  let running = 0;
  let success = 0;
  let fail = 0;
  let skipped = 0;
  for (const node of nodes) {
    const s = String(node.status || "idle").toLowerCase();
    if (s === "running") running += 1;
    else if (s === "success" || s === "ok" || s === "done") success += 1;
    else if (s === "error" || s === "fail") fail += 1;
    else if (s === "skipped") skipped += 1;
  }
  graphStatsEl.textContent = `Nodes ${nodes.length} · Running ${running} · Done ${success} · Fail ${fail} · Skipped ${skipped}`;
  if (fail > 0) setPillState(graphStatsPillEl, "error");
  else if (running > 0) setPillState(graphStatsPillEl, "running");
  else if (success > 0) setPillState(graphStatsPillEl, "ok");
  else setPillState(graphStatsPillEl, "idle");
}

function renderEventLog() {
  if (!eventsEl) return;
  const q = String((eventFilterEl && eventFilterEl.value) || "").trim().toLowerCase();
  const filtered = q ? eventLines.filter((line) => line.toLowerCase().includes(q)) : eventLines;
  const view = filtered.slice(-1000);
  eventsEl.textContent = view.join("\n");
  eventsEl.scrollTop = eventsEl.scrollHeight;
}

function logEvent(line) {
  eventLines.push(String(line || ""));
  renderEventLog();
}

function clearEvents() {
  eventLines = [];
  renderEventLog();
}

function getFieldValue(id) {
  const el = document.getElementById(id);
  if (!el) return "";
  return String(el.value || "").trim();
}

function toNodeId(node) {
  return String((node && (node.node_id || node.id)) || "");
}

function findNode(nodeId) {
  if (!currentGraph || !Array.isArray(currentGraph.nodes)) return null;
  return currentGraph.nodes.find((n) => toNodeId(n) === nodeId) || null;
}

function isImagePath(path) {
  const p = String(path || "").toLowerCase();
  return p.endsWith(".png") || p.endsWith(".jpg") || p.endsWith(".jpeg") || p.endsWith(".gif") || p.endsWith(".webp");
}

function isReportMarkdownPath(path) {
  const p = String(path || "").replaceAll("\\", "/").toLowerCase();
  return p.endsWith("/report/clinical_report.md") || p.endsWith("clinical_report.md") || p.endsWith("/report/report.md") || p.endsWith("report.md") || p.endsWith("final_report.md");
}

function resetReport() {
  reportPath = "";
  if (renderReportBtnEl) renderReportBtnEl.disabled = true;
  if (reportMetaEl) reportMetaEl.textContent = "Waiting for report artifact.";
  if (reportViewerEl) reportViewerEl.textContent = "No report rendered yet.";
  if (openReportRawLinkEl) openReportRawLinkEl.href = "#";
}

function setReportPath(path) {
  reportPath = String(path || "");
  if (renderReportBtnEl) renderReportBtnEl.disabled = !reportPath;
  if (reportMetaEl) reportMetaEl.textContent = reportPath ? `Found report: ${reportPath}` : "Waiting for report artifact.";
  if (openReportRawLinkEl && reportPath && lastRunId) {
    openReportRawLinkEl.href = `/runs/${lastRunId}/artifact?path=${encodeURIComponent(reportPath)}`;
  }
}

function buildGraphUrl() {
  const params = new URLSearchParams();
  params.set("domain", getFieldValue("domain") || "prostate");
  params.set("request_type", getFieldValue("requestType") || "full_pipeline");
  const caseId = getFieldValue("caseId");
  const caseRef = getFieldValue("dicomDir");
  if (caseId) params.set("case_id", caseId);
  if (caseRef) params.set("case_ref", caseRef);
  return `/graph?${params.toString()}`;
}

function mergeNodeWithOverride(node) {
  if (!node) return null;
  const nodeId = toNodeId(node);
  const ov = localNodeOverrides[nodeId] || {};
  const merged = { ...node };
  if (ov.tool_locked !== undefined) merged.tool_locked = ov.tool_locked;
  if (ov.skip !== undefined) {
    merged.meta = { ...(merged.meta || {}), skip: Boolean(ov.skip) };
  }
  if (ov.config_values && typeof ov.config_values === "object") {
    merged.config_values = { ...(merged.config_values || {}), ...ov.config_values };
  }
  return merged;
}

function applyLocalOverridesToGraph() {
  if (!currentGraph || !Array.isArray(currentGraph.nodes)) return;
  currentGraph.nodes = currentGraph.nodes.map((n) => mergeNodeWithOverride(n));
}

async function checkServer() {
  const baseUrl = getFieldValue("serverBaseUrl") || "http://127.0.0.1:8000";
  setServerStatus(`Checking ${baseUrl} ...`, "running");
  const params = new URLSearchParams({ base_url: baseUrl });
  const res = await fetch(`/server/probe?${params.toString()}`);
  if (!res.ok) {
    setServerStatus(`Check failed (${res.status})`, "error");
    return null;
  }
  const data = await res.json();
  if (data.ok) {
    setServerStatus(`Reachable at ${data.base_url}`, "ok");
  } else {
    const checks = Array.isArray(data.checks) ? data.checks : [];
    const firstErr = checks.find((x) => !x.ok && x.error);
    setServerStatus(firstErr ? String(firstErr.error) : `Unreachable: ${data.base_url}`, "error");
  }
  return data;
}

function layoutForNode(node, graphLayout) {
  const id = toNodeId(node);
  if (graphLayout && graphLayout[id]) return graphLayout[id];
  return { x: 0, y: 0 };
}

function styleClassFromStatus(status) {
  const s = String(status || "idle").toLowerCase();
  if (s === "running") return "status-running";
  if (s === "success" || s === "ok" || s === "done") return "status-success";
  if (s === "error" || s === "fail") return "status-error";
  if (s === "skipped") return "status-skipped";
  if (s === "blocked") return "status-blocked";
  return "status-idle";
}

function captureCurrentLayout() {
  if (!cy || !currentGraph || !Array.isArray(currentGraph.nodes)) return;
  const out = {};
  for (const node of currentGraph.nodes) {
    const id = toNodeId(node);
    const ele = cy.getElementById(id);
    if (!ele || ele.empty()) continue;
    out[id] = { x: ele.position("x"), y: ele.position("y") };
  }
  currentGraph.layout = out;
}

function fitGraph(animate = true) {
  if (!cy) return;
  if (animate) {
    cy.animate({
      fit: { padding: 36 },
      duration: 320,
      easing: "ease-out-cubic",
    });
  } else {
    cy.fit(undefined, 36);
  }
}

function autoLayoutGraph() {
  if (!cy) return;
  cy.layout({
    name: "breadthfirst",
    directed: true,
    padding: 34,
    spacingFactor: 1.28,
    animate: true,
    animationDuration: 420,
  }).run();
  setTimeout(captureCurrentLayout, 450);
}

function focusSelectedNode() {
  if (!cy || !selectedNodeId) return;
  const ele = cy.getElementById(selectedNodeId);
  if (!ele || ele.empty()) return;
  cy.animate({
    center: { eles: ele },
    zoom: Math.max(cy.zoom(), 1.06),
    duration: 260,
  });
}

function renderGraph() {
  if (!currentGraph) return;
  const nodes = (currentGraph.nodes || []).map((n) => {
    const id = toNodeId(n);
    const statusClass = styleClassFromStatus(n.status);
    const cls = `${statusClass}${id === selectedNodeId ? " selected" : ""}`;
    return {
      data: {
        id,
        label: n.label || id,
        node_kind: n.node_kind || n.kind || "tool",
      },
      position: layoutForNode(n, currentGraph.layout || {}),
      classes: cls,
    };
  });
  const edges = (currentGraph.edges || []).map((e, i) => ({
    data: {
      id: `e${i}`,
      source: String(e.source || ""),
      target: String(e.target || ""),
      label: String(e.condition || ""),
    },
  }));

  if (cy) cy.destroy();
  cy = cytoscape({
    container: document.getElementById("graph"),
    elements: [...nodes, ...edges],
    style: [
      {
        selector: "node",
        style: {
          shape: "round-rectangle",
          width: 176,
          height: 62,
          "background-color": "#1a6a86",
          color: "#ffffff",
          label: "data(label)",
          "font-size": 11,
          "font-weight": 600,
          "text-wrap": "wrap",
          "text-max-width": "154px",
          "text-valign": "center",
          "border-width": 1,
          "border-color": "#dce7f2",
          "overlay-opacity": 0,
        },
      },
      { selector: "node.status-running", style: { "background-color": "#d97a1b" } },
      { selector: "node.status-success", style: { "background-color": "#2e8b57" } },
      { selector: "node.status-error", style: { "background-color": "#c54444" } },
      { selector: "node.status-skipped", style: { "background-color": "#657f94" } },
      { selector: "node.status-blocked", style: { "background-color": "#a25a2b" } },
      {
        selector: "node.selected",
        style: {
          "border-width": 4,
          "border-color": "#f7c15c",
        },
      },
      {
        selector: "edge",
        style: {
          width: 2,
          "line-color": "#7b95ad",
          "target-arrow-color": "#7b95ad",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          label: "data(label)",
          "font-size": 9,
          color: "#2f4357",
          "text-background-color": "#ffffff",
          "text-background-opacity": 0.95,
          "text-background-padding": "2px",
        },
      },
    ],
    layout: { name: "preset" },
  });

  cy.on("tap", "node", (evt) => {
    selectedNodeId = evt.target.id();
    renderInspector();
    refreshNodeClasses();
  });

  cy.on("dragfree", "node", () => {
    captureCurrentLayout();
  });

  updateGraphStats();
  setTimeout(() => fitGraph(false), 40);
}

function refreshNodeClasses() {
  if (!cy || !currentGraph) return;
  for (const node of currentGraph.nodes || []) {
    const id = toNodeId(node);
    const n = cy.getElementById(id);
    if (!n || n.empty()) continue;
    n.classes(`${styleClassFromStatus(node.status)}${id === selectedNodeId ? " selected" : ""}`);
  }
}

async function loadGraph() {
  const res = await fetch(buildGraphUrl());
  if (!res.ok) throw new Error(`graph request failed (${res.status})`);
  currentGraph = await res.json();
  applyLocalOverridesToGraph();
  renderGraph();
  if (selectedNodeId) renderInspector();
  updateGraphStats();
}

function readJsonFieldValue(raw, schema) {
  const t = String((schema && schema.type) || "").toLowerCase();
  if (t === "number") {
    const n = Number(raw);
    return Number.isFinite(n) ? n : 0;
  }
  if (t === "integer") {
    const n = Number.parseInt(raw, 10);
    return Number.isFinite(n) ? n : 0;
  }
  if (t === "boolean") {
    return raw === "true";
  }
  if (t === "object" || t === "array") {
    try {
      return JSON.parse(raw || (t === "array" ? "[]" : "{}"));
    } catch {
      return t === "array" ? [] : {};
    }
  }
  return raw;
}

function renderConfigEditor(schema, values) {
  configEditorEl.innerHTML = "";
  const props = (schema && schema.properties && typeof schema.properties === "object") ? schema.properties : {};
  const keys = Object.keys(props);
  if (!keys.length) {
    const div = document.createElement("div");
    div.className = "hint";
    div.textContent = "No editable config fields.";
    configEditorEl.appendChild(div);
    return;
  }

  for (const key of keys) {
    const fs = props[key] || {};
    const row = document.createElement("div");
    row.className = "config-row";
    const label = document.createElement("label");
    label.textContent = key;
    row.appendChild(label);
    let input;
    const type = String(fs.type || "").toLowerCase();
    const val = values && key in values ? values[key] : (fs.default !== undefined ? fs.default : "");
    if (Array.isArray(fs.enum) && fs.enum.length) {
      input = document.createElement("select");
      for (const opt of fs.enum) {
        const o = document.createElement("option");
        o.value = String(opt);
        o.textContent = String(opt);
        if (String(opt) === String(val)) o.selected = true;
        input.appendChild(o);
      }
    } else if (type === "boolean") {
      input = document.createElement("select");
      for (const opt of ["true", "false"]) {
        const o = document.createElement("option");
        o.value = opt;
        o.textContent = opt;
        if (String(Boolean(val)) === opt) o.selected = true;
        input.appendChild(o);
      }
    } else if (type === "object" || type === "array") {
      input = document.createElement("textarea");
      input.value = JSON.stringify(val || (type === "array" ? [] : {}), null, 2);
    } else {
      input = document.createElement("input");
      input.value = String(val ?? "");
    }
    input.dataset.cfgKey = key;
    input.dataset.cfgType = type || "string";
    row.appendChild(input);
    if (fs.description) {
      const hint = document.createElement("small");
      hint.textContent = String(fs.description);
      row.appendChild(hint);
    }
    configEditorEl.appendChild(row);
  }
}

function collectConfigValues(schema) {
  const out = {};
  const props = (schema && schema.properties && typeof schema.properties === "object") ? schema.properties : {};
  const fields = configEditorEl.querySelectorAll("[data-cfg-key]");
  for (const el of fields) {
    const key = el.dataset.cfgKey;
    const fs = props[key] || {};
    out[key] = readJsonFieldValue(String(el.value || ""), fs);
  }
  return out;
}

function renderInspector() {
  const node = mergeNodeWithOverride(findNode(selectedNodeId));
  if (!node) {
    inspectorNodeNameEl.textContent = "Select a node";
    nodeRationaleEl.textContent = "No node selected.";
    toolCandidatesEl.innerHTML = "";
    agentSelectedToolEl.value = "";
    lockToolToggleEl.checked = false;
    skipNodeToggleEl.checked = false;
    configEditorEl.innerHTML = "";
    return;
  }
  inspectorNodeNameEl.textContent = `${toNodeId(node)} | ${node.label || ""}`;
  const selected = String(node.tool_selected || node.tool_name || "");
  const locked = String(node.tool_locked || "");
  const candidates = Array.isArray(node.tool_candidates) ? node.tool_candidates : [];
  agentSelectedToolEl.value = selected;
  toolCandidatesEl.innerHTML = "";
  for (const t of candidates) {
    const opt = document.createElement("option");
    opt.value = String(t);
    opt.textContent = String(t);
    if ((locked && locked === t) || (!locked && selected === t)) opt.selected = true;
    toolCandidatesEl.appendChild(opt);
  }
  if (!candidates.length && selected) {
    const opt = document.createElement("option");
    opt.value = selected;
    opt.textContent = selected;
    opt.selected = true;
    toolCandidatesEl.appendChild(opt);
  }
  lockToolToggleEl.checked = Boolean(locked);
  skipNodeToggleEl.checked = Boolean(node.meta && node.meta.skip);
  nodeRationaleEl.textContent = String((node.meta && node.meta.selection_rationale) || "No rationale.");
  renderConfigEditor(node.config_schema || { type: "object", properties: {} }, node.config_values || {});
}

function updateNodeStatus(nodeId, status) {
  if (!currentGraph || !Array.isArray(currentGraph.nodes)) return;
  for (const n of currentGraph.nodes) {
    if (toNodeId(n) === nodeId) {
      n.status = status;
      break;
    }
  }
  refreshNodeClasses();
  updateGraphStats();
}

function renderArtifactList() {
  if (!artifactsEl) return;
  artifactsEl.innerHTML = "";
  const q = String((artifactFilterEl && artifactFilterEl.value) || "").trim().toLowerCase();
  const list = q ? artifactPaths.filter((p) => p.toLowerCase().includes(q)) : artifactPaths;
  for (const path of list) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = `/runs/${lastRunId}/artifact?path=${encodeURIComponent(path)}`;
    a.target = "_blank";
    a.textContent = path;
    li.appendChild(a);
    artifactsEl.appendChild(li);
  }
}

function upsertArtifact(path) {
  if (!path) return;
  if (isReportMarkdownPath(path)) {
    const shouldAutoRender = !reportPath;
    setReportPath(path);
    if (shouldAutoRender) renderReportMarkdown().catch((err) => logEvent(`report render failed: ${err}`));
    return;
  }
  if (!artifactPaths.includes(path)) {
    artifactPaths.push(path);
    renderArtifactList();
  }
}

function renderEvidence(items) {
  evidenceEl.innerHTML = "";
  if (!Array.isArray(items) || !items.length) {
    evidenceEl.textContent = "No evidence yet.";
    return;
  }
  for (const rec of items) {
    const card = document.createElement("div");
    card.className = "evidence-card";
    const url = String(rec.url || "");
    if (isImagePath(rec.path || "")) {
      const img = document.createElement("img");
      img.src = url;
      img.alt = String(rec.caption || rec.path || "evidence");
      card.appendChild(img);
    }
    const cap = document.createElement("div");
    cap.className = "evidence-cap";
    const a = document.createElement("a");
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = String(rec.caption || rec.path || "evidence");
    cap.appendChild(a);
    card.appendChild(cap);
    evidenceEl.appendChild(card);
  }
}

async function renderReportMarkdown() {
  if (!lastRunId) return;
  const res = await fetch(`/runs/${lastRunId}/report`);
  if (!res.ok) {
    reportMetaEl.textContent = `Failed to load report (${res.status})`;
    return;
  }
  const data = await res.json();
  const md = String(data.markdown || "");
  if (data.report_path) setReportPath(data.report_path);
  if (md) {
    if (window.marked && typeof window.marked.parse === "function") {
      reportViewerEl.innerHTML = window.marked.parse(md);
    } else {
      reportViewerEl.textContent = md;
    }
    reportMetaEl.textContent = `Rendered: ${data.report_path || "clinical_report.md"}`;
  } else {
    reportViewerEl.textContent = "No report rendered yet.";
    reportMetaEl.textContent = "Report not found.";
  }
  renderEvidence(data.evidence || []);
}

function eventSummary(evt) {
  const t = String(evt.event_type || "event");
  const n = String(evt.node_id || "");
  const tool = String(evt.tool_name || "");
  return `${t}${n ? ` ${n}` : ""}${tool ? ` ${tool}` : ""}`;
}

function connectEvents(runId, replay = false) {
  if (evtSource) evtSource.close();
  const url = replay ? `/runs/${runId}/events?replay=true&speed=1.5` : `/runs/${runId}/events`;
  evtSource = new EventSource(url);
  evtSource.onmessage = (msg) => {
    const evt = JSON.parse(msg.data);
    const type = String(evt.event_type || "");
    logEvent(eventSummary(evt));

    if (type === "node_start") updateNodeStatus(String(evt.node_id || ""), "running");
    if (type === "node_end") {
      const st = String(evt.status || "").toLowerCase();
      updateNodeStatus(String(evt.node_id || ""), st === "ok" || st === "done" ? "success" : "error");
    }
    if (type === "artifact" && evt.artifact && evt.artifact.path) {
      upsertArtifact(String(evt.artifact.path));
    }
    if (type === "run_end") {
      setRunStatus(`Run ${runId}: completed`, "ok");
      renderReportMarkdown().catch((err) => logEvent(`report fetch failed: ${err}`));
    }
    if (type === "run_error") {
      setRunStatus(`Run ${runId}: failed`, "error");
    }
    if (type === "stream_end") {
      setRunStatus(`Run ${runId}: stream finished`, "ok");
      evtSource.close();
    }
  };
  evtSource.onerror = () => {
    setRunStatus(`Run ${runId}: event stream disconnected`, "error");
  };
}

async function applyNodePatch() {
  if (!selectedNodeId) return;
  const node = findNode(selectedNodeId);
  if (!node) return;
  const locked = lockToolToggleEl.checked ? String(toolCandidatesEl.value || "").trim() : "";
  const skip = Boolean(skipNodeToggleEl.checked);
  const cfgValues = collectConfigValues(node.config_schema || { type: "object", properties: {} });
  const ov = {
    tool_locked: locked,
    config_values: cfgValues,
    skip,
  };
  localNodeOverrides[selectedNodeId] = ov;

  if (lastRunId) {
    const res = await fetch(`/runs/${lastRunId}/patch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_overrides: { [selectedNodeId]: ov }, replan: false }),
    });
    if (res.ok) {
      const data = await res.json();
      currentGraph = data.graph || currentGraph;
      renderGraph();
      renderInspector();
      logEvent(`patch applied to ${selectedNodeId}`);
      return;
    }
    logEvent(`patch failed (${res.status})`);
  }

  applyLocalOverridesToGraph();
  renderGraph();
  renderInspector();
}

function restoreDefaults() {
  if (!selectedNodeId) return;
  delete localNodeOverrides[selectedNodeId];
  if (currentGraph) {
    loadGraph().catch((err) => logEvent(`graph reload failed: ${err}`));
  }
}

async function rerunFromNode() {
  if (!lastRunId || !selectedNodeId) return;
  setRunStatus(`Rerun from ${selectedNodeId}...`, "running");
  const res = await fetch(`/runs/${lastRunId}/rerun`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ start_from_node_id: selectedNodeId, invalidate_downstream: true }),
  });
  if (!res.ok) {
    setRunStatus(`Rerun failed (${res.status})`, "error");
    return;
  }
  setRunStatus(`Run ${lastRunId}: rerun started`, "running");
  connectEvents(lastRunId, false);
}

async function startRun() {
  clearEvents();
  artifactPaths = [];
  renderArtifactList();
  evidenceEl.textContent = "No evidence yet.";
  resetReport();
  await loadGraph().catch((err) => logEvent(`graph load failed: ${err}`));
  if ((getFieldValue("llmMode") || "server") === "server") {
    await checkServer().catch((err) => setServerStatus(`Server check error: ${err}`, "error"));
  }
  const payload = {
    case_id: getFieldValue("caseId") || "demo_case",
    dicom_case_dir: getFieldValue("dicomDir") || null,
    domain: getFieldValue("domain") || "prostate",
    request_type: getFieldValue("requestType") || "full_pipeline",
    llm_mode: getFieldValue("llmMode") || "server",
    server_base_url: getFieldValue("serverBaseUrl") || "http://127.0.0.1:8000",
    server_model: getFieldValue("serverModel") || "google/medgemma-1.5-4b-it",
    engine: "shell",
    node_overrides: localNodeOverrides,
  };
  setRunStatus("Starting run...", "running");
  const res = await fetch("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    setRunStatus(`Run start failed (${res.status})`, "error");
    return;
  }
  const data = await res.json();
  lastRunId = data.run_id;
  setRunStatus(`Run ${lastRunId}: started`, "running");
  connectEvents(lastRunId, false);
}

function setActiveTab(tabId) {
  const panels = document.querySelectorAll(".tab-panel");
  const buttons = document.querySelectorAll(".tab-btn");
  for (const p of panels) {
    p.classList.toggle("active", p.id === tabId);
  }
  for (const b of buttons) {
    b.classList.toggle("active", b.dataset.tab === tabId);
  }
}

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  for (const b of buttons) {
    b.addEventListener("click", () => {
      const tab = String(b.dataset.tab || "").trim();
      if (tab) setActiveTab(tab);
    });
  }
}

document.getElementById("reloadGraphBtn").addEventListener("click", () => {
  loadGraph().catch((err) => logEvent(`graph load failed: ${err}`));
});
document.getElementById("startRunBtn").addEventListener("click", () => {
  startRun().catch((err) => logEvent(`start run failed: ${err}`));
});
document.getElementById("replayBtn").addEventListener("click", () => {
  if (!lastRunId) return;
  clearEvents();
  connectEvents(lastRunId, true);
});
document.getElementById("checkServerBtn").addEventListener("click", () => {
  checkServer().catch((err) => setServerStatus(`Server check error: ${err}`, "error"));
});
document.getElementById("renderReportBtn").addEventListener("click", () => {
  renderReportMarkdown().catch((err) => logEvent(`report render failed: ${err}`));
});
document.getElementById("applyNodeBtn").addEventListener("click", () => {
  applyNodePatch().catch((err) => logEvent(`patch failed: ${err}`));
});
document.getElementById("restoreDefaultsBtn").addEventListener("click", restoreDefaults);
document.getElementById("rerunNodeBtn").addEventListener("click", () => {
  rerunFromNode().catch((err) => logEvent(`rerun failed: ${err}`));
});

document.getElementById("fitGraphBtn").addEventListener("click", () => {
  fitGraph(true);
});
document.getElementById("autoLayoutBtn").addEventListener("click", () => {
  autoLayoutGraph();
});
document.getElementById("focusNodeBtn").addEventListener("click", () => {
  focusSelectedNode();
});

if (eventFilterEl) {
  eventFilterEl.addEventListener("input", renderEventLog);
}
if (artifactFilterEl) {
  artifactFilterEl.addEventListener("input", renderArtifactList);
}

["domain", "requestType", "caseId", "dicomDir"].forEach((id) => {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener("change", () => {
    loadGraph().catch((err) => logEvent(`graph load failed: ${err}`));
  });
});

window.addEventListener("keydown", (evt) => {
  if (evt.key === "Enter" && (evt.ctrlKey || evt.metaKey)) {
    evt.preventDefault();
    startRun().catch((err) => logEvent(`start run failed: ${err}`));
  }
  if (evt.key.toLowerCase() === "f" && evt.shiftKey) {
    evt.preventDefault();
    fitGraph(true);
  }
});

initTabs();
setActiveTab("inspectorTab");
setRunStatus("Idle", "idle");
setServerStatus("Not checked", "idle");
loadGraph().catch((err) => logEvent(`graph load failed: ${err}`));
resetReport();
checkServer().catch((err) => setServerStatus(`Server check error: ${err}`, "error"));
