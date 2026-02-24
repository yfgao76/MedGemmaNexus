# Graph GUI

This implementation follows a hybrid design:
- Graph view is derived from shell planner DAG (`mri_agent_shell/...`) for interactive demo topology.
- Execution remains `agent/langgraph/loop.py` via `run_langgraph_agent`.
- Runtime events are streamed from real run logs (`events.jsonl` or `execution_log.jsonl`).

## Success Criteria
- `GET /graph` returns GraphJSON for node/edge rendering.
- Runs stream node/tool/artifact events over SSE.
- Run bundle export includes trace + artifacts for replay demos.

## Backend
- Server + run/event adapter: `ui/graph_gui_server.py`
- Graph schema tools: `runtime/graphjson.py`
- Planner/DAG source: `mri_agent_shell/agent/brain.py`, `core/plan_dag.py`
- Runtime runner: `agent/langgraph/loop.py`

## API
- `GET /graph`
- `GET /graph/mermaid`
- `GET /server/probe?base_url=...`
- `POST /runs`
- `GET /runs/{id}`
- `GET /runs/{id}/events` (SSE; supports `?replay=true&speed=...`)
- `GET /runs/{id}/artifacts`
- `GET /runs/{id}/artifact?path=...`
- `GET /runs/{id}/bundle`

## Frontend
- Static UI: `ui/static/index.html`
- Styles: `ui/static/styles.css`
- Runtime logic: `ui/static/app.js`

## Run
```bash
uvicorn MRI_Agent.ui.graph_gui_server:app --host 0.0.0.0 --port 8787 --reload
```

By default, GUI run requests use:
- `llm_mode=server`
- `server_model=google/medgemma-1.5-4b-it`

Run panel fields also include:
- `server_base_url` (default `http://127.0.0.1:8000`)
- `server_model`
- `Check Server` button (calls `/server/probe`)
