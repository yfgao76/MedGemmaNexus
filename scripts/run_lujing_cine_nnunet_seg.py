#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path("/home/longz2/common/medgemma")
CMR_REVERSE_ROOT = Path("/home/longz2/common/cmr_reverse")
INPUT_NII = Path("/home/longz2/common/cmr_reverse/lujing_data/exported_from_twix.nii.gz")
OUTPUT_DIR = Path("/home/longz2/common/cmr_reverse/lujing_data/cine_seg_nnunet")
NNUNET_PYTHON = Path("/home/longz2/common/cmr_reverse/revimg/bin/python")


def _ensure_import_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def _check_paths() -> None:
    if not INPUT_NII.is_file():
        raise FileNotFoundError(f"Input NIfTI not found: {INPUT_NII}")
    if not NNUNET_PYTHON.exists():
        raise FileNotFoundError(f"nnUNet python not found: {NNUNET_PYTHON}")
    if not (CMR_REVERSE_ROOT / "nnunetdata" / "results").exists():
        raise FileNotFoundError(f"nnUNet results folder missing: {CMR_REVERSE_ROOT / 'nnunetdata' / 'results'}")


def _matrix_info() -> dict:
    import SimpleITK as sitk  # type: ignore

    img = sitk.ReadImage(str(INPUT_NII))
    return {
        "input_path": str(INPUT_NII),
        "dimension": int(img.GetDimension()),
        "matrix_size": [int(x) for x in img.GetSize()],
        "spacing": [float(x) for x in img.GetSpacing()],
    }


def _write_matrix_info(info: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "input_matrix_info.txt"
    lines = [
        f"input_path: {info['input_path']}",
        f"dimension: {info['dimension']}",
        f"matrix_size: {tuple(info['matrix_size'])}",
        f"spacing: {tuple(info['spacing'])}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _check_nnunet_python_venv() -> None:
    # Use abspath, not resolve(), to preserve the venv entrypoint path.
    py = Path(os.path.abspath(os.path.expanduser(str(NNUNET_PYTHON))))
    cmd = [
        str(py),
        "-c",
        "import sys, torch; print(sys.executable); print(torch.__version__)",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "nnUNet python cannot import torch.\n"
            f"python: {py}\n"
            f"stderr:\n{proc.stderr.strip()}"
        )
    print("[INFO] nnUNet python check:")
    print(proc.stdout.strip())


def main() -> None:
    _ensure_import_path()
    _check_paths()
    _check_nnunet_python_venv()

    from MRI_Agent.tools.cardiac_cine_segmentation import run_cardiac_cine_segmentation  # noqa: WPS433

    info = _matrix_info()
    matrix_info_path = _write_matrix_info(info)
    print(f"[INFO] dimension: {info['dimension']}")
    print(f"[INFO] matrix_size: {tuple(info['matrix_size'])}")
    print(f"[INFO] spacing: {tuple(info['spacing'])}")
    print(f"[INFO] matrix info saved: {matrix_info_path}")

    # Keep symlink entrypoint semantics for venv python.
    nnunet_python = Path(os.path.abspath(os.path.expanduser(str(NNUNET_PYTHON))))
    results_folder = (CMR_REVERSE_ROOT / "nnunetdata" / "results").resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = run_cardiac_cine_segmentation(
        cine_path=INPUT_NII.resolve(),
        output_dir=OUTPUT_DIR.resolve(),
        cmr_reverse_root=CMR_REVERSE_ROOT.resolve(),
        nnunet_python=nnunet_python,
        results_folder=results_folder,
        task_name="Task900_ACDC_Phys",
        trainer_class_name="nnUNetTrainerV2_InvGreAug",
        model="2d",
        folds=[0, 1, 2, 3, 4],
        checkpoint="model_final_checkpoint",
        disable_tta=True,
        num_threads_preprocessing=2,
        num_threads_nifti_save=2,
    )

    result_json = OUTPUT_DIR / "cardiac_cine_seg_result.json"
    result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] Result JSON: {result_json}")
    print(f"[OK] Primary segmentation: {result.get('seg_path')}")


if __name__ == "__main__":
    main()
