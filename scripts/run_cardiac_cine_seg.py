#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_import_path() -> None:
    here = Path(__file__).resolve()
    # .../MRI_Agent/scripts -> add workspace root so `import MRI_Agent` works.
    ws_root = here.parents[2]
    if str(ws_root) not in sys.path:
        sys.path.insert(0, str(ws_root))


def main() -> None:
    _ensure_import_path()
    from MRI_Agent.tools.cardiac_cine_segmentation import run_cardiac_cine_segmentation  # noqa: WPS433

    ap = argparse.ArgumentParser(description="Run cardiac cine segmentation via cmr_reverse nnUNet weights.")
    ap.add_argument("--cine-path", required=True, help="Cine NIfTI path (.nii.gz/.nii) or directory containing cine NIfTI(s).")
    ap.add_argument("--output-dir", required=True, help="Output directory for predictions and masks.")
    ap.add_argument("--cmr-reverse-root", default="/home/longz2/common/cmr_reverse", help="cmr_reverse repository root.")
    ap.add_argument("--nnunet-python", default="", help="Python executable with nnUNet installed (default: <cmr_reverse_root>/revimg/bin/python).")
    ap.add_argument("--results-folder", default="", help="nnUNet RESULTS_FOLDER (default: <cmr_reverse_root>/nnunetdata/results).")
    ap.add_argument("--task-name", default="Task900_ACDC_Phys")
    ap.add_argument("--trainer-class-name", default="nnUNetTrainerV2_InvGreAug")
    ap.add_argument("--model", default="2d")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--checkpoint", default="model_final_checkpoint")
    ap.add_argument("--disable-tta", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num-threads-preprocessing", type=int, default=2)
    ap.add_argument("--num-threads-nifti-save", type=int, default=2)
    args = ap.parse_args()

    cmr_root = Path(args.cmr_reverse_root).expanduser().resolve()
    # Keep the provided interpreter path without resolving symlinks. Resolving can
    # collapse a venv entrypoint (e.g. revimg/bin/python -> /usr/bin/python3.11),
    # which breaks env-specific packages such as torch/nnUNet.
    nnunet_python = (
        Path(os.path.abspath(os.path.expanduser(args.nnunet_python)))
        if args.nnunet_python
        else (cmr_root / "revimg" / "bin" / "python")
    )
    results_folder = Path(args.results_folder).expanduser().resolve() if args.results_folder else (cmr_root / "nnunetdata" / "results")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_cardiac_cine_segmentation(
        cine_path=Path(args.cine_path).expanduser().resolve(),
        output_dir=out_dir,
        cmr_reverse_root=cmr_root,
        nnunet_python=nnunet_python,
        results_folder=results_folder,
        task_name=str(args.task_name),
        trainer_class_name=str(args.trainer_class_name),
        model=str(args.model),
        folds=[int(x) for x in args.folds],
        checkpoint=str(args.checkpoint),
        disable_tta=bool(args.disable_tta),
        num_threads_preprocessing=int(args.num_threads_preprocessing),
        num_threads_nifti_save=int(args.num_threads_nifti_save),
    )

    result_json = out_dir / "cardiac_cine_seg_result.json"
    result_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] Result JSON: {result_json}")
    print(f"[OK] Primary segmentation: {result.get('seg_path')}")


if __name__ == "__main__":
    main()
