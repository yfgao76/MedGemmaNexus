# MedGemma On Compute Node (No `srun`)

This avoids running MedGemma on the login node and fixes `Connection refused` when GUI uses `llm_mode=server`.

## 1) Submit MedGemma server to Slurm

```bash
cd /common/lidxxlab/Yifan/MedGemmaChallenge
MODEL_PATH=/common/lidxxlab/Yifan/MedGemmaChallenge/models/medgemma-1.5-4b-it \
PARTITION=gpu \
GRES=gpu:1 \
bash MRI_Agent/scripts/submit_medgemma_server_slurm.sh
```

It prints `JOB_ID`.

## 2) Wait until job is running and get node

```bash
squeue -j <JOB_ID> -o '%.18i %.9T %.20M %.30N'
```

When `ST=R`, note `NODELIST` (for example `esplhpc-cp090`).

## 3) Connect from login node to compute-node server

Option A (recommended): keep GUI default `http://127.0.0.1:8000` by opening a tunnel.

```bash
bash MRI_Agent/scripts/open_medgemma_tunnel.sh --job-id <JOB_ID> --local-port 8000 --remote-port 8000
```

Then verify:

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/health
```

Option B: no tunnel, use compute node URL directly in GUI:

- `server_base_url = http://<NODELIST>:8000`

## 4) Start GUI server and run

```bash
cd /common/lidxxlab/Yifan/MedGemmaChallenge
python -m uvicorn MRI_Agent.ui.graph_gui_server:app --host 0.0.0.0 --port 8787
```

In GUI:
- `llm_mode = server`
- `server_base_url = http://127.0.0.1:8000` (if tunnel) or `http://<NODELIST>:8000`
- `server_model = google/medgemma-1.5-4b-it`
- Click `Check Server` first, then `Start Run`.

## 5) If still refused

- Confirm tunnel terminal is still alive.
- Confirm Slurm job is still running.
- Check logs:
  - `MRI_Agent/runs/medgemma_server/medgemma_srv_<JOB_ID>.out`
  - `MRI_Agent/runs/medgemma_server/medgemma_srv_<JOB_ID>.err`
