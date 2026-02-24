#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_GROUPS = ["NOR", "MINF", "DCM", "HCM", "RV"]


def _patient_num(name: str) -> int:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else 0


def _parse_info_cfg(info_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not info_path.exists():
        return out
    for line in info_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip()
        val = v.strip()
        if key:
            out[key] = val
    return out


def _grouped_patients(training_root: Path, groups: List[str]) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {g: [] for g in groups}
    wanted = {g.upper() for g in groups}
    for p in sorted(training_root.glob("patient*"), key=lambda x: _patient_num(x.name)):
        if not p.is_dir():
            continue
        info = _parse_info_cfg(p / "Info.cfg")
        grp = str(info.get("Group") or "").strip().upper()
        if grp in wanted:
            grouped.setdefault(grp, []).append(p)
    return grouped


def _copy_patient(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            return
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _resolve_frame_path(patient_dir: Path, frame_idx: int) -> Path:
    return patient_dir / f"{patient_dir.name}_frame{int(frame_idx):02d}.nii.gz"


def _build_manifest_entries(selected: List[Tuple[str, Path]], copied_patients_root: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for group, src_patient_dir in selected:
        patient_id = src_patient_dir.name
        dst_patient_dir = copied_patients_root / patient_id
        info = _parse_info_cfg(dst_patient_dir / "Info.cfg")
        ed = int(str(info.get("ED") or "0") or 0)
        es = int(str(info.get("ES") or "0") or 0)
        ed_path = _resolve_frame_path(dst_patient_dir, ed)
        es_path = _resolve_frame_path(dst_patient_dir, es)
        entry = {
            "group": group,
            "patient_id": patient_id,
            "ed_frame": ed,
            "es_frame": es,
            "patient_dir": str(dst_patient_dir),
            "cine_4d_path": str(dst_patient_dir / f"{patient_id}_4d.nii.gz"),
            "ed_frame_path": str(ed_path),
            "es_frame_path": str(es_path),
        }
        if not ed_path.exists() or not es_path.exists():
            raise FileNotFoundError(f"Missing ED/ES frame for {patient_id}: ED={ed_path.exists()}, ES={es_path.exists()}")
        entries.append(entry)
    return entries


def _write_run_script(
    *,
    script_path: Path,
    manifest_entries: List[Dict[str, object]],
    cmr_reverse_root: Path,
    export_script: Path,
    out_root: Path,
) -> None:
    npz_dir = out_root / "derived_t1_molli" / "reverse_npz"
    nifti_dir = out_root / "derived_t1_molli" / "t1_molli_nifti"

    lines: List[str] = []
    lines.append("#!/usr/bin/env bash")
    lines.append("set -euo pipefail")
    lines.append("")
    lines.append(f'CMR_REVERSE_ROOT="${{CMR_REVERSE_ROOT:-{cmr_reverse_root}}}"')
    lines.append('CMR_REVERSE_PY="${CMR_REVERSE_PY:-$CMR_REVERSE_ROOT/revimg/bin/python}"')
    lines.append('DEVICE="${DEVICE:-cuda}"')
    lines.append('NUM_STEPS="${NUM_STEPS:-200}"')
    lines.append('BATCH_SIZE="${BATCH_SIZE:-5}"')
    lines.append('STEP_SIZE="${STEP_SIZE:-400}"')
    lines.append('FA="${FA:-55.0}"')
    lines.append('IN_SCALE="${IN_SCALE:-0.3}"')
    lines.append('TR="${TR:-47.58}"')
    lines.append('TE="${TE:-1.64}"')
    lines.append("")
    lines.append(f'NPZ_DIR="{npz_dir}"')
    lines.append(f'NIFTI_DIR="{nifti_dir}"')
    lines.append('mkdir -p "$NPZ_DIR" "$NIFTI_DIR"')
    lines.append("")
    lines.append('cd "$CMR_REVERSE_ROOT"')
    lines.append("")

    for entry in manifest_entries:
        patient_id = str(entry["patient_id"])
        group = str(entry["group"])
        ed_path = str(entry["ed_frame_path"])
        es_path = str(entry["es_frame_path"])
        for phase, frame_path in (("ed", ed_path), ("es", es_path)):
            npz_path = f"$NPZ_DIR/{patient_id}_{phase}.npz"
            prefix = f"{patient_id}_{phase}"
            lines.append(f'echo "[RUN] {group} {patient_id} {phase.upper()}"')
            lines.append(
                '"$CMR_REVERSE_PY" "$CMR_REVERSE_ROOT/reverse_single_cine.py" '
                f'--input "{frame_path}" --output "{npz_path}" '
                '--device "$DEVICE" --num_steps "$NUM_STEPS" --batch_size "$BATCH_SIZE" '
                '--step_size "$STEP_SIZE" --fa "$FA" --in_scale "$IN_SCALE" --tr "$TR" --te "$TE"'
            )
            lines.append(
                '"$CMR_REVERSE_PY" '
                f'"{export_script}" '
                f'--npz "{npz_path}" --reference-nifti "{frame_path}" '
                f'--out-dir "$NIFTI_DIR" --prefix "{prefix}" --export-molli-series'
            )
            lines.append("")
    lines.append('echo "[DONE] Reverse cine ED/ES -> T1 MOLLI outputs completed."')
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare ACDC demo cohort and emit cmr_reverse commands for ED/ES -> T1 MOLLI conversion.")
    ap.add_argument("--acdc-training-root", default="/common/yanghjlab/Data/raw_acdc_dataset/training")
    ap.add_argument("--demo-root", default=str((Path(__file__).resolve().parents[1] / "demo").resolve()))
    ap.add_argument("--groups", default=",".join(DEFAULT_GROUPS), help="Comma-separated groups")
    ap.add_argument("--subjects-per-group", type=int, default=4, help="How many subjects to pick from each group")
    ap.add_argument("--output-name", default="acdc_test_group_4_per_group")
    ap.add_argument("--cmr-reverse-root", default="/home/longz2/common/cmr_reverse")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing copied patient folders")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    export_script = (repo_root / "MRI_Agent" / "scripts" / "export_t1_molli_from_reverse_npz.py").resolve()
    training_root = Path(args.acdc_training_root).expanduser().resolve()
    demo_root = Path(args.demo_root).expanduser().resolve()
    groups = [g.strip().upper() for g in str(args.groups).split(",") if g.strip()]
    n_per_group = int(args.subjects_per_group)

    if n_per_group <= 0:
        raise ValueError("--subjects-per-group must be > 0")
    if not training_root.exists():
        raise FileNotFoundError(f"ACDC training root not found: {training_root}")

    grouped = _grouped_patients(training_root, groups)
    selected: List[Tuple[str, Path]] = []
    for g in groups:
        cands = grouped.get(g, [])
        if len(cands) < n_per_group:
            raise RuntimeError(f"Group {g} has only {len(cands)} subjects, need {n_per_group}")
        selected.extend([(g, p) for p in cands[:n_per_group]])

    out_root = demo_root / "cases" / str(args.output_name)
    patients_root = out_root / "patients"
    patients_root.mkdir(parents=True, exist_ok=True)

    for group, src in selected:
        dst = patients_root / src.name
        _copy_patient(src, dst, overwrite=bool(args.overwrite))

    manifest = _build_manifest_entries(selected, patients_root)
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "acdc_training_root": str(training_root),
                "groups": groups,
                "subjects_per_group": n_per_group,
                "total_subjects": len(manifest),
                "entries": manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    run_script_path = out_root / "run_reverse_to_t1_molli.sh"
    _write_run_script(
        script_path=run_script_path,
        manifest_entries=manifest,
        cmr_reverse_root=Path(args.cmr_reverse_root).expanduser(),
        export_script=export_script,
        out_root=out_root,
    )

    print(f"[OK] Prepared cohort under: {out_root}")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Run script: {run_script_path}")
    print(f"[INFO] Selected {len(manifest)} subjects from groups={groups}, n_per_group={n_per_group}")


if __name__ == "__main__":
    main()
