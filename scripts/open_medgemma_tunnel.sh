#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Open SSH tunnel on login node so local port forwards to MedGemma server on compute node.

Usage:
  bash MRI_Agent/scripts/open_medgemma_tunnel.sh --node <compute-node> [--local-port 8000] [--remote-port 8000]
  bash MRI_Agent/scripts/open_medgemma_tunnel.sh --job-id <slurm-job-id> [--local-port 8000] [--remote-port 8000]

Examples:
  bash MRI_Agent/scripts/open_medgemma_tunnel.sh --node esplhpc-cp090
  bash MRI_Agent/scripts/open_medgemma_tunnel.sh --job-id 1234567

Then from login node:
  curl http://127.0.0.1:8000/v1/models
USAGE
}

NODE=""
JOB_ID=""
LOCAL_PORT="8000"
REMOTE_PORT="8000"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node)
      NODE="${2:-}"
      shift 2
      ;;
    --job-id)
      JOB_ID="${2:-}"
      shift 2
      ;;
    --local-port)
      LOCAL_PORT="${2:-8000}"
      shift 2
      ;;
    --remote-port)
      REMOTE_PORT="${2:-8000}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${NODE}" && -n "${JOB_ID}" ]]; then
  NODE="$(squeue -h -j "${JOB_ID}" -o "%N" | head -n 1 | tr -d '[:space:]')"
fi

if [[ -z "${NODE}" || "${NODE}" == "(null)" || "${NODE}" == "None" ]]; then
  echo "[ERROR] Could not resolve compute node. Provide --node or a RUNNING --job-id." >&2
  exit 2
fi

echo "[INFO] Forwarding login-node 127.0.0.1:${LOCAL_PORT} -> ${NODE}:127.0.0.1:${REMOTE_PORT}"
echo "[INFO] Keep this terminal open. Ctrl-C to stop tunnel."
exec ssh -N -o ExitOnForwardFailure=yes -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "${USER}@${NODE}"
