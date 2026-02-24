#!/usr/bin/env bash
set -euo pipefail

# Submit MedGemma OpenAI-compatible server to a GPU compute node (no srun needed).
#
# Example:
#   MODEL_PATH=/common/lidxxlab/Yifan/MedGemmaChallenge/models/medgemma-1.5-4b-it \
#   PARTITION=gpu GRES=gpu:1 \
#   bash MRI_Agent/scripts/submit_medgemma_server_slurm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/medgemma_server}"
mkdir -p "${RUN_DIR}"

PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

MODEL_PATH="${MODEL_PATH:-/common/lidxxlab/Yifan/MedGemmaChallenge/models/medgemma-1.5-4b-it}"
CONDA_SH="${CONDA_SH:-/apps/miniconda/23.11.0-2/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-mriagent}"

TS="$(date +%Y%m%d_%H%M%S)"
JOB_SCRIPT="${RUN_DIR}/medgemma_server_${TS}.sbatch"

cat > "${JOB_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH -J medgemma_srv
#SBATCH -p ${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH -c ${CPUS}
#SBATCH --mem=${MEM}
#SBATCH -t ${TIME_LIMIT}
#SBATCH -o ${RUN_DIR}/medgemma_srv_%j.out
#SBATCH -e ${RUN_DIR}/medgemma_srv_%j.err

set -euo pipefail
echo "[INFO] job_id=\${SLURM_JOB_ID} host=\$(hostname -f) start=\$(date -Is)"
if [ -f "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

export HF_HUB_OFFLINE="\${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="\${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

python "${ROOT_DIR}/scripts/medgemma_openai_server.py" \\
  --model "${MODEL_PATH}" \\
  --host "${HOST}" \\
  --port "${PORT}" \\
  --dtype "${DTYPE}" \\
  --device-map "${DEVICE_MAP}"
EOF

JOB_ID="$(sbatch "${JOB_SCRIPT}" | awk '{print $4}')"
echo "[OK] Submitted MedGemma server job: ${JOB_ID}"
echo "[INFO] Track status:"
echo "  squeue -j ${JOB_ID} -o '%.18i %.9T %.20M %.30N'"
echo "[INFO] Once RUNNING, either:"
echo "  1) Set GUI server_base_url to http://<NODE>:${PORT}"
echo "  2) Keep GUI default and open tunnel:"
echo "     bash ${ROOT_DIR}/scripts/open_medgemma_tunnel.sh --job-id ${JOB_ID} --local-port ${PORT} --remote-port ${PORT}"
echo "[INFO] Logs:"
echo "  ${RUN_DIR}/medgemma_srv_${JOB_ID}.out"
echo "  ${RUN_DIR}/medgemma_srv_${JOB_ID}.err"
