#!/usr/bin/env bash
set -euo pipefail

# Fixed paths for this case.
PROJECT_ROOT="/home/longz2/common/medgemma"
CMR_REVERSE_ROOT="/home/longz2/common/cmr_reverse"
INPUT_NII="/home/longz2/common/cmr_reverse/lujing_data/exported_from_twix.nii.gz"
OUTPUT_DIR="/home/longz2/common/cmr_reverse/lujing_data/cine_seg_nnunet"

# Entrypoints.
SEG_RUNNER="${PROJECT_ROOT}/MRI_Agent/scripts/run_cardiac_cine_seg.py"
NNUNET_PYTHON="${CMR_REVERSE_ROOT}/revimg/bin/python"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${INPUT_NII}" ]]; then
  echo "[ERROR] Input file not found: ${INPUT_NII}" >&2
  exit 1
fi

if [[ ! -x "${NNUNET_PYTHON}" ]]; then
  echo "[ERROR] nnUNet python not executable: ${NNUNET_PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${SEG_RUNNER}" ]]; then
  echo "[ERROR] Segmentation runner not found: ${SEG_RUNNER}" >&2
  exit 1
fi

MATRIX_INFO_TXT="${OUTPUT_DIR}/input_matrix_info.txt"
echo "[INFO] Checking input matrix size..."
"${NNUNET_PYTHON}" - "${INPUT_NII}" "${MATRIX_INFO_TXT}" <<'PY'
import sys
from pathlib import Path
import SimpleITK as sitk

inp = Path(sys.argv[1]).expanduser().resolve()
out_txt = Path(sys.argv[2]).expanduser().resolve()
img = sitk.ReadImage(str(inp))
size = tuple(int(x) for x in img.GetSize())  # x, y, z (, t)
spacing = tuple(float(x) for x in img.GetSpacing())
dim = int(img.GetDimension())

lines = [
    f"input_path: {inp}",
    f"dimension: {dim}",
    f"matrix_size: {size}",
    f"spacing: {spacing}",
]
text = "\n".join(lines) + "\n"
print(text, end="")
out_txt.write_text(text, encoding="utf-8")
PY

echo "[INFO] Running nnUNet cine segmentation..."
"${NNUNET_PYTHON}" "${SEG_RUNNER}" \
  --cine-path "${INPUT_NII}" \
  --output-dir "${OUTPUT_DIR}" \
  --cmr-reverse-root "${CMR_REVERSE_ROOT}" \
  --nnunet-python "${NNUNET_PYTHON}" \
  --results-folder "${CMR_REVERSE_ROOT}/nnunetdata/results" \
  --task-name "Task900_ACDC_Phys" \
  --trainer-class-name "nnUNetTrainerV2_InvGreAug" \
  --model "2d" \
  --folds 0 1 2 3 4 \
  --checkpoint "model_final_checkpoint" \
  --disable-tta \
  --num-threads-preprocessing 2 \
  --num-threads-nifti-save 2

echo "[DONE] Output directory: ${OUTPUT_DIR}"
echo "[DONE] Matrix info: ${MATRIX_INFO_TXT}"
echo "[DONE] Result JSON: ${OUTPUT_DIR}/cardiac_cine_seg_result.json"
