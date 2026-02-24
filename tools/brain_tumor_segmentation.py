"""
Tool: brats_mri_segmentation

Purpose:
- Run MONAI brats_mri_segmentation bundle on 4-channel aligned brain MRI.
- Produce tumor subregion masks with 3 channels: TC, WT, ET.

Inputs:
- t1c_path: NIfTI path for T1c
- t1_path: NIfTI path for T1
- t2_path: NIfTI path for T2
- flair_path: NIfTI path for FLAIR
- bundle_root (optional): MONAI bundle root dir
- device (optional): "auto" | "cuda" | "cpu"
- output_subdir (optional)

Outputs:
- seg_path: raw bundle label map (labels 1/2/4)
- tumor_subregions_path: 3-channel mask (C=3) [0=TC,1=WT,2=ET]
- tc_mask_path, wt_mask_path, et_mask_path: individual binary masks
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from shutil import move
from typing import Any, Dict, List, Optional, Tuple

from ..commands.registry import Tool
from ..commands.schemas import ArtifactRef, ToolContext, ToolSpec

DEFAULT_BUNDLE_ROOT = Path("~/.cache/torch/hub/bundle/brats_mri_segmentation").expanduser()

BRATS_SPEC = ToolSpec(
    name="brats_mri_segmentation",
    description=(
        "Run MONAI brats_mri_segmentation on aligned 1x1x1 mm brain MRI (T1c, T1, T2, FLAIR). "
        "Outputs tumor subregion masks: channel 0=TC (tumor core), channel 1=WT (whole tumor), "
        "channel 2=ET (enhancing tumor)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "t1c_path": {"type": "string"},
            "t1_path": {"type": "string"},
            "t2_path": {"type": "string"},
            "flair_path": {"type": "string"},
            "bundle_root": {"type": "string"},
            "device": {"type": "string", "enum": ["auto", "cpu", "cuda"]},
            "output_subdir": {"type": "string"},
        },
        "required": ["t1c_path", "t1_path", "t2_path", "flair_path"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "seg_path": {"type": "string"},
            "tumor_subregions_path": {"type": "string"},
            "tc_mask_path": {"type": "string"},
            "wt_mask_path": {"type": "string"},
            "et_mask_path": {"type": "string"},
            "note": {"type": "string"},
        },
        "required": ["seg_path", "tumor_subregions_path"],
    },
    version="0.1.0",
    tags=["segmentation", "brats", "brain", "monai"],
)


def _require_libs():
    try:
        import torch  # type: ignore
        import numpy as np  # type: ignore
        import nibabel as nib  # type: ignore
        from monai.bundle import create_workflow  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies for brats_mri_segmentation: torch, monai, nibabel, numpy. "
            "Install via: pip install torch monai nibabel numpy"
        ) from e
    return torch, np, nib, create_workflow


def _resolve_paths(args: Dict[str, Any]) -> Tuple[Path, Path, Path, Path]:
    t1c = Path(args["t1c_path"]).expanduser().resolve()
    t1 = Path(args["t1_path"]).expanduser().resolve()
    t2 = Path(args["t2_path"]).expanduser().resolve()
    flair = Path(args["flair_path"]).expanduser().resolve()
    for p in (t1c, t1, t2, flair):
        if not p.exists():
            raise FileNotFoundError(f"Missing input NIfTI: {p}")
    return t1c, t1, t2, flair


def _bundle_files(bundle_root: Path) -> Tuple[Path, Path, Path]:
    inference_json = bundle_root / "configs" / "inference.json"
    logging_conf = bundle_root / "configs" / "logging.conf"
    metadata_json = bundle_root / "configs" / "metadata.json"
    if not inference_json.exists():
        raise FileNotFoundError(
            f"MONAI BRATS bundle not found at {bundle_root}. "
            "Run: python -m monai.bundle download brats_mri_segmentation --version 0.5.2 --bundle_dir ~/.cache/torch/hub/bundle"
        )
    return inference_json, logging_conf, metadata_json


def _prepare_config_for_device(inference_json: Path, device: str) -> Tuple[str, Optional[str]]:
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be auto|cpu|cuda (got {device})")

    torch, _, _, _ = _require_libs()
    use_cpu = device == "cpu" or (device == "auto" and not torch.cuda.is_available())
    if not use_cpu:
        return str(inference_json), None

    with open(inference_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["device"] = "cpu"
    cfg.setdefault("checkpointloader", {})["map_location"] = "cpu"

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, tmp)
    tmp.flush()
    return tmp.name, tmp.name


def _run_bundle(images: List[str], bundle_root: Path, device: str, out_dir: Path) -> Path:
    _, _, _, create_workflow = _require_libs()
    inference_json, logging_conf, metadata_json = _bundle_files(bundle_root)

    config_to_use, temp_cfg = _prepare_config_for_device(inference_json, device)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            workflow = create_workflow(
                workflow_type="infer",
                bundle_root=str(bundle_root),
                config_file=str(config_to_use),
                logging_file=str(logging_conf),
                meta_file=str(metadata_json),
                test_datalist=[{"image": images}],
                output_dtype="uint8",
                separate_folder=False,
                output_ext=".nii.gz",
                output_dir=temp_dir,
            )
            workflow.evaluator.run()
            output_file = Path(temp_dir) / sorted(Path(temp_dir).iterdir())[0].name
            out_dir.mkdir(parents=True, exist_ok=True)
            seg_path = out_dir / "brats_segmentation.nii.gz"
            move(str(output_file), str(seg_path))
            return seg_path
    finally:
        if temp_cfg:
            try:
                Path(temp_cfg).unlink(missing_ok=True)
            except Exception:
                pass


def _build_subregion_masks(seg_path: Path, out_dir: Path) -> Dict[str, Path]:
    _, np, nib, _ = _require_libs()
    seg_img = nib.load(str(seg_path))
    seg = seg_img.get_fdata()
    if seg.ndim != 3:
        seg = np.squeeze(seg)

    tc = np.logical_or(seg == 1, seg == 4)
    wt = np.logical_or(tc, seg == 2)
    et = seg == 4

    tc_path = out_dir / "brats_tc_mask.nii.gz"
    wt_path = out_dir / "brats_wt_mask.nii.gz"
    et_path = out_dir / "brats_et_mask.nii.gz"
    multi_path = out_dir / "brats_tumor_subregions_c3.nii.gz"

    nib.save(nib.Nifti1Image(tc.astype("uint8"), seg_img.affine, seg_img.header), str(tc_path))
    nib.save(nib.Nifti1Image(wt.astype("uint8"), seg_img.affine, seg_img.header), str(wt_path))
    nib.save(nib.Nifti1Image(et.astype("uint8"), seg_img.affine, seg_img.header), str(et_path))

    stack = np.stack([tc, wt, et], axis=0).astype("uint8")
    nib.save(nib.Nifti1Image(stack, seg_img.affine, seg_img.header), str(multi_path))

    return {
        "tc": tc_path,
        "wt": wt_path,
        "et": et_path,
        "multi": multi_path,
    }


def brats_mri_segmentation(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    _require_libs()

    t1c, t1, t2, flair = _resolve_paths(args)
    bundle_root = Path(args.get("bundle_root") or DEFAULT_BUNDLE_ROOT).expanduser().resolve()
    device = str(args.get("device") or "auto").lower()

    output_subdir = args.get("output_subdir") or "brats_mri_segmentation"
    out_dir = ctx.artifacts_dir / output_subdir

    seg_path = _run_bundle([str(t1c), str(t1), str(t2), str(flair)], bundle_root, device, out_dir)
    masks = _build_subregion_masks(seg_path, out_dir)

    warnings: List[str] = []
    try:
        import numpy as np  # type: ignore
        import nibabel as nib  # type: ignore

        seg_vals = np.unique(np.asarray(nib.load(str(seg_path)).get_fdata()))
        if seg_vals.size <= 1:
            warnings.append("BRATS segmentation produced an empty mask (all background).")
    except Exception:
        pass

    note = "Tumor subregions in brats_tumor_subregions_c3.nii.gz with channels [0=TC,1=WT,2=ET]."

    return {
        "data": {
            "seg_path": str(seg_path),
            "tumor_subregions_path": str(masks["multi"]),
            "tc_mask_path": str(masks["tc"]),
            "wt_mask_path": str(masks["wt"]),
            "et_mask_path": str(masks["et"]),
            "note": note,
        },
        "artifacts": [
            ArtifactRef(path=str(seg_path), kind="nifti", description="BRATS raw label map (1/2/4)"),
            ArtifactRef(path=str(masks["multi"]), kind="nifti", description="BRATS subregions (C=3, 0=TC,1=WT,2=ET)"),
            ArtifactRef(path=str(masks["tc"]), kind="nifti", description="Tumor core (TC) binary mask"),
            ArtifactRef(path=str(masks["wt"]), kind="nifti", description="Whole tumor (WT) binary mask"),
            ArtifactRef(path=str(masks["et"]), kind="nifti", description="Enhancing tumor (ET) binary mask"),
        ],
        "warnings": warnings,
    }


def build_tool() -> Tool:
    return Tool(spec=BRATS_SPEC, func=brats_mri_segmentation)
