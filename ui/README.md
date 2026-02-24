# MRI Agent Interactive Studio

This guide covers how to run and use the web UI (`graph_gui_server.py`) with MedGemma as the backend model server.

## 1) Current status (as of 2026-02-23)

- UI server: running on `esplhpccompbio-lv03.cm.cluster` (`172.21.0.14`) at `127.0.0.1:8787`.
- MedGemma Slurm job (current session): `3754188` on `esplhpc-cp075` (OpenAI-compatible server on port `8000`).

Check live state:

```bash
squeue -j 3754188 -o '%.18i %.9T %.20M %.30N'
curl http://esplhpc-cp075:8000/health
curl http://esplhpc-cp075:8000/v1/models
```

## 2) Launch MedGemma server on compute node

From repo root:

```bash
cd /common/lidxxlab/Yifan/MedGemmaChallenge
MODEL_PATH=/common/lidxxlab/Yifan/MedGemmaChallenge/models/medgemma-1.5-4b-it \
PARTITION=gpu \
GRES=gpu:1 \
CPUS=8 \
MEM=64G \
TIME_LIMIT=08:00:00 \
bash MRI_Agent/scripts/submit_medgemma_server_slurm.sh
```

Notes:
- Use `gpu` partition by default.
- If all GPU nodes are unavailable, job will remain pending until resources free up.
- Do not switch to `preemptable` unless explicitly requested.

Logs:

```bash
ls -1t /common/lidxxlab/Yifan/MedGemmaChallenge/MRI_Agent/runs/medgemma_server | head
# then inspect
# .../medgemma_srv_<JOBID>.out
# .../medgemma_srv_<JOBID>.err
```

## 3) Launch / relaunch Interactive Studio server

```bash
cd /common/lidxxlab/Yifan/MedGemmaChallenge
setsid /home/gaoy4/.conda/envs/mriagent/bin/python -m uvicorn \
  MRI_Agent.ui.graph_gui_server:app \
  --host 127.0.0.1 \
  --port 8787 \
  > MRI_Agent/runs/gui_server/ui_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

Health check:

```bash
curl http://127.0.0.1:8787/health
```

## 4) Open UI in your local browser (SSH tunnel)

From your laptop:

```bash
ssh -N -L 8787:127.0.0.1:8787 gaoy4@172.21.0.14
```

Then open:

```text
http://127.0.0.1:8787
```

## 5) Configure MedGemma in the UI

In the control dock:

- `LLM Mode`: `server`
- `Server Base URL`:
  - `http://<compute-node>:8000` (direct), or
  - `http://127.0.0.1:8000` if you opened `open_medgemma_tunnel.sh`
- `Server Model`: `/common/lidxxlab/Yifan/MedGemmaChallenge/models/medgemma-1.5-4b-it`

Use **Check Server** before **Start Run**.

## 6) Interactive Studio workflow

1. Set `Case ID`, `Domain`, `Request Type`, and optional `Case/DICOM Dir`.
2. Click **Reload Graph** to inspect DAG structure.
3. Click nodes to open **Inspector**:
   - view agent tool choice/candidates
   - lock tool
   - edit node config
   - skip node
4. Click **Apply Node Patch** (stores per-run patch) or **Rerun From Node**.
5. Click **Start Run** for full execution.
6. Watch progress in:
   - status pills
   - graph node colors
   - **Events** tab (with filter)
7. Review outputs in:
   - **Report** tab (renders `artifacts/report/clinical_report.md` when available)
   - **Evidence** tab (image gallery)
   - **Artifacts** tab (all files)
8. Use **Replay Last Run** for SSE replay.

## 7) Graph interaction shortcuts

- `Ctrl/Cmd + Enter`: start run
- `Shift + F`: fit graph to viewport
- Buttons: `Fit Graph`, `Auto Layout`, `Focus Selected`

## 8) Useful API endpoints

- `GET /health`
- `GET /graph`
- `POST /runs`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/events`
- `GET /runs/{run_id}/report`
- `GET /runs/{run_id}/artifacts`

## 9) Troubleshooting

### `ECONNREFUSED 127.0.0.1:8787`
- UI server not running on login node.
- Relaunch section 3 and re-check `/health`.

### `Connection refused` when checking model server
- MedGemma job not yet `RUNNING`.
- Wrong compute node in `Server Base URL`.
- Port `8000` not reachable from current host.

### MedGemma job stays pending
- Cluster GPU nodes unavailable/down.
- Check reason:

```bash
scontrol show job <JOBID> | rg 'JobState=|Reason=|Partition=|NodeList='
```

### UI loads but run fails immediately
- Verify `Case/DICOM Dir` exists and readable.
- Confirm domain/request type matches case modality.
- Check run event stream and run artifacts for exact failing node.

## 10) Stop services

Stop UI server:

```bash
pkill -f 'uvicorn MRI_Agent.ui.graph_gui_server:app --host 127.0.0.1 --port 8787'
```

Cancel MedGemma job:

```bash
scancel <JOBID>
```
