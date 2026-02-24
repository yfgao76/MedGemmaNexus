#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


GROUPS_DEFAULT = ["NOR", "MINF", "DCM", "HCM", "RV"]


def _utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ensure_import_path() -> None:
    here = Path(__file__).resolve()
    ws_root = here.parents[2]
    if str(ws_root) not in sys.path:
        sys.path.insert(0, str(ws_root))


def _parse_info_cfg(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            key = k.strip()
            val = v.strip()
            if key:
                out[key] = val
    except Exception:
        return {}
    return out


def _patient_num_from_name(name: str) -> int:
    m = re.search(r"(\d+)$", str(name))
    return int(m.group(1)) if m else 0


def _find_training_patients(acdc_root: Path, split: str, groups: List[str]) -> Dict[str, List[Path]]:
    split_dir = acdc_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"ACDC split dir not found: {split_dir}")
    wanted = {g.upper() for g in groups}
    grouped: Dict[str, List[Path]] = {g: [] for g in groups}
    for patient_dir in sorted(split_dir.glob("patient*"), key=lambda p: _patient_num_from_name(p.name)):
        if not patient_dir.is_dir():
            continue
        info = _parse_info_cfg(patient_dir / "Info.cfg")
        grp = str(info.get("Group") or "").strip().upper()
        if not grp or grp not in wanted:
            continue
        grouped.setdefault(grp, []).append(patient_dir)
    return grouped


def _sample_group_cases(grouped: Dict[str, List[Path]], n_per_group: int, seed: int, random_sample: bool) -> Dict[str, List[Path]]:
    rng = random.Random(int(seed))
    out: Dict[str, List[Path]] = {}
    for grp, items in grouped.items():
        if n_per_group <= 0 or len(items) <= n_per_group:
            out[grp] = list(items)
            continue
        out[grp] = sorted(rng.sample(items, k=int(n_per_group)), key=lambda p: _patient_num_from_name(p.name)) if random_sample else list(items[:n_per_group])
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _parse_agent_run_dir(stdout: str, stderr: str, case_runs_root: Path, case_id: str) -> Optional[Path]:
    text = "\n".join([stdout or "", stderr or ""])
    patterns = [
        r"Agent loop run output:\s*(\S+)",
        r"LangGraph agent run output:\s*(\S+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            p = Path(m.group(1)).expanduser().resolve()
            if p.exists():
                return p
    case_dir = case_runs_root / case_id
    if not case_dir.exists():
        return None
    runs = [p for p in case_dir.iterdir() if p.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _load_agent_outputs(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "predicted_group": None,
        "ground_truth_group": None,
        "ground_truth_match": None,
        "classification_path": None,
        "llm_error": None,
        "status": "unknown",
    }
    cls_path = run_dir / "artifacts" / "classification" / "cardiac_cine" / "cardiac_cine_classification.json"
    if cls_path.exists():
        cls = _read_json(cls_path)
        out["classification_path"] = str(cls_path)
        out["predicted_group"] = cls.get("predicted_group")
        out["ground_truth_group"] = cls.get("ground_truth_group")
        out["ground_truth_match"] = cls.get("ground_truth_match")
        out["metrics"] = cls.get("metrics") if isinstance(cls.get("metrics"), dict) else {}
        out["status"] = "ok"
    final_json = run_dir / "artifacts" / "final" / "final_report.json"
    if final_json.exists():
        try:
            final_obj = _read_json(final_json).get("final_report", {})
            if isinstance(final_obj, dict):
                if final_obj.get("error"):
                    out["llm_error"] = str(final_obj.get("error"))
                    if out["status"] != "ok":
                        out["status"] = "completed_without_llm"
                if final_obj.get("status") and out["status"] == "unknown":
                    out["status"] = str(final_obj.get("status"))
        except Exception:
            pass
    return out


def _extract_metrics_metrics(obj: Dict[str, Any]) -> Dict[str, Any]:
    metrics = obj.get("metrics") if isinstance(obj.get("metrics"), dict) else {}
    return {
        "lv_ef_percent": _safe_float(metrics.get("lv_ef_percent")),
        "rv_ef_percent": _safe_float(metrics.get("rv_ef_percent")),
        "lv_edv_ml": _safe_float(metrics.get("lv_edv_ml")),
        "rv_edv_ml": _safe_float(metrics.get("rv_edv_ml")),
        "max_myo_thickness_mm": _safe_float(metrics.get("max_myo_thickness_mm")),
    }


def _run_agent_case(
    *,
    python_exe: str,
    case_id: str,
    patient_dir: Path,
    case_runs_root: Path,
    llm_mode: str,
    server_base_url: str,
    server_model: str,
    api_model: str,
    api_base_url: str,
    max_steps: int,
    autofix_mode: str,
) -> Dict[str, Any]:
    cmd: List[str] = [
        str(python_exe),
        "-m",
        "MRI_Agent.agent.loop",
        "--case-id",
        str(case_id),
        "--dicom-case-dir",
        str(patient_dir),
        "--domain",
        "cardiac",
        "--llm-mode",
        str(llm_mode),
        "--autofix-mode",
        str(autofix_mode),
        "--max-steps",
        str(int(max_steps)),
        "--runs-root",
        str(case_runs_root),
    ]
    if llm_mode == "server":
        cmd.extend(["--server-base-url", str(server_base_url), "--server-model", str(server_model)])
    if llm_mode in {"openai", "anthropic", "gemini"}:
        if api_model:
            cmd.extend(["--api-model", str(api_model)])
        if api_base_url:
            cmd.extend(["--api-base-url", str(api_base_url)])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    run_dir = _parse_agent_run_dir(proc.stdout or "", proc.stderr or "", case_runs_root, case_id)
    result: Dict[str, Any] = {
        "mode": "agent",
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-30:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-30:]),
    }
    if run_dir is not None:
        result.update(_load_agent_outputs(run_dir))
    else:
        result["status"] = "failed"
        result["error"] = "Could not resolve run_dir from command output."
    if proc.returncode != 0 and result.get("status") == "ok":
        result["status"] = "partial_ok_nonzero_exit"
    return result


def _resolve_cine_path(patient_dir: Path) -> Path:
    cands = sorted(patient_dir.glob("*_4d.nii.gz"))
    if cands:
        return cands[0]
    cands = sorted(patient_dir.glob("*.nii.gz"))
    if cands:
        return cands[0]
    raise FileNotFoundError(f"No NIfTI found under {patient_dir}")


def _extract_ed_es_cases(case_results: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    ed_case = None
    es_case = None
    for item in case_results:
        cid = str(item.get("case_id") or "").lower()
        if cid.endswith("_ed"):
            ed_case = item
        elif cid.endswith("_es"):
            es_case = item
    return ed_case, es_case


def _run_toolchain_case(
    *,
    case_id: str,
    patient_dir: Path,
    run_root: Path,
    cmr_reverse_root: Path,
    nnunet_python: str,
    results_folder: str,
    with_features: bool,
) -> Dict[str, Any]:
    _ensure_import_path()
    from MRI_Agent.commands.schemas import ToolContext  # noqa: WPS433
    from MRI_Agent.tools.cardiac_cine_classification import classify_cardiac_cine_disease  # noqa: WPS433
    from MRI_Agent.tools.cardiac_cine_segmentation import run_cardiac_cine_segmentation  # noqa: WPS433
    from MRI_Agent.tools.roi_features import extract_roi_features  # noqa: WPS433

    run_dir = run_root / case_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(
        case_id=case_id,
        run_id="manual",
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        case_state_path=run_dir / "case_state.json",
    )

    info_cfg = patient_dir / "Info.cfg"
    cine_path = _resolve_cine_path(patient_dir)
    nnunet_py = Path(nnunet_python).expanduser().resolve() if nnunet_python else (cmr_reverse_root / "revimg" / "bin" / "python")
    results_dir = Path(results_folder).expanduser().resolve() if results_folder else (cmr_reverse_root / "nnunetdata" / "results")

    seg = run_cardiac_cine_segmentation(
        cine_path=cine_path,
        output_dir=artifacts_dir / "segmentation" / "cardiac_cine",
        cmr_reverse_root=cmr_reverse_root,
        nnunet_python=nnunet_py,
        results_folder=results_dir,
        use_ed_es_only=True,
        acdc_info_cfg=info_cfg if info_cfg.exists() else None,
    )
    case_results = list(seg.get("case_results") or [])
    ed_case, es_case = _extract_ed_es_cases(case_results)
    cls_args: Dict[str, Any] = {
        "seg_path": str(ed_case.get("seg_path") if isinstance(ed_case, dict) else seg.get("seg_path")),
        "cine_path": str(cine_path),
        "patient_info_path": str(info_cfg),
        "output_subdir": "classification/cardiac_cine",
    }
    if isinstance(ed_case, dict) and ed_case.get("seg_path"):
        cls_args["ed_seg_path"] = str(ed_case["seg_path"])
    if isinstance(es_case, dict) and es_case.get("seg_path"):
        cls_args["es_seg_path"] = str(es_case["seg_path"])
    cls = classify_cardiac_cine_disease(cls_args, ctx)
    data = cls.get("data", {}) if isinstance(cls, dict) else {}

    feature_paths: Dict[str, str] = {}
    if with_features:
        roi_masks: List[Dict[str, str]] = []
        images: List[str] = []
        input_dir = Path(str(seg.get("input_dir") or "")).expanduser().resolve()
        for item in case_results:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("case_id") or "")
            phase = None
            if cid.lower().endswith("_ed"):
                phase = "ED"
            elif cid.lower().endswith("_es"):
                phase = "ES"
            if phase is None:
                continue
            img_p = input_dir / f"{cid}_0000.nii.gz"
            if img_p.exists():
                images.append(str(img_p))
            for key, name in (("rv_mask_path", "RV"), ("myo_mask_path", "MYO"), ("lv_mask_path", "LV")):
                p = item.get(key)
                if p:
                    roi_masks.append({"name": f"{name}_{phase}", "path": str(p)})
        if images and roi_masks:
            feat = extract_roi_features(
                {
                    "images": images,
                    "roi_masks": roi_masks,
                    "roi_interpretation": "cardiac",
                    "radiomics_mode": "auto",
                    "output_subdir": "features/cardiac_texture",
                },
                ctx,
            )
            feat_data = feat.get("data", {}) if isinstance(feat, dict) else {}
            for k in ("feature_table_path", "texture_radiomics_summary_path", "texture_radiomics_heatmap_path"):
                v = feat_data.get(k)
                if v:
                    feature_paths[k] = str(v)

    out = {
        "mode": "toolchain",
        "run_dir": str(run_dir),
        "status": "ok",
        "predicted_group": data.get("predicted_group"),
        "ground_truth_group": data.get("ground_truth_group"),
        "ground_truth_match": data.get("ground_truth_match"),
        "classification_path": data.get("classification_path"),
        "metrics": data.get("metrics") if isinstance(data.get("metrics"), dict) else {},
        "feature_outputs": feature_paths,
        "llm_error": None,
    }
    return out


def _write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    headers = [
        "patient_id",
        "group",
        "mode",
        "status",
        "predicted_group",
        "ground_truth_group",
        "ground_truth_match",
        "lv_ef_percent",
        "rv_ef_percent",
        "lv_edv_ml",
        "rv_edv_ml",
        "max_myo_thickness_mm",
        "run_dir",
        "classification_path",
        "llm_error",
        "error",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in headers})


def _summarize(rows: List[Dict[str, Any]], groups: List[str]) -> Dict[str, Any]:
    by_group: Dict[str, Dict[str, Any]] = {}
    for g in groups:
        g_rows = [r for r in rows if str(r.get("group")) == g]
        valid = [r for r in g_rows if isinstance(r.get("ground_truth_match"), bool)]
        n_ok = sum(1 for r in valid if bool(r.get("ground_truth_match")))
        by_group[g] = {
            "n_total": len(g_rows),
            "n_with_gt_match": len(valid),
            "n_correct": n_ok,
            "accuracy": (float(n_ok) / len(valid) if valid else None),
            "n_failed": sum(1 for r in g_rows if str(r.get("status")) not in {"ok", "completed_without_llm"}),
        }
    all_valid = [r for r in rows if isinstance(r.get("ground_truth_match"), bool)]
    all_ok = sum(1 for r in all_valid if bool(r.get("ground_truth_match")))
    return {
        "n_cases": len(rows),
        "n_with_gt_match": len(all_valid),
        "n_correct": all_ok,
        "accuracy": (float(all_ok) / len(all_valid) if all_valid else None),
        "by_group": by_group,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="ACDC cardiac multi-disease batch regression (NOR/MINF/DCM/HCM/RV).")
    ap.add_argument("--acdc-root", default="/common/yanghjlab/Data/raw_acdc_dataset")
    ap.add_argument("--split", default="training")
    ap.add_argument("--groups", default="NOR,MINF,DCM,HCM,RV")
    ap.add_argument("--n-per-group", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random-sample", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mode", choices=["agent", "toolchain"], default="agent")
    ap.add_argument("--python-exe", default=sys.executable, help="Python executable for launching agent mode.")
    ap.add_argument("--runs-root", default="", help="Batch output root. Default: MRI_Agent/runs/acdc_batch_regression/<timestamp>")
    ap.add_argument("--case-prefix", default="acdc_batch")
    ap.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)

    # Agent mode knobs.
    ap.add_argument("--llm-mode", choices=["server", "stub", "openai", "anthropic", "gemini"], default="server")
    ap.add_argument("--server-base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--server-model", default="Qwen/Qwen3-VL-30B-A3B-Thinking")
    ap.add_argument("--api-model", default="")
    ap.add_argument("--api-base-url", default="")
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--autofix-mode", choices=["force", "coach", "off"], default="force")

    # Toolchain mode knobs.
    ap.add_argument("--cmr-reverse-root", default="/common/longz2/cmr_reverse")
    ap.add_argument("--nnunet-python", default="/home/longz2/.conda/envs/qwen_vllm/bin/python")
    ap.add_argument("--results-folder", default="/common/longz2/cmr_reverse/nnunetdata/results")
    ap.add_argument("--with-features", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    groups = [x.strip().upper() for x in str(args.groups).split(",") if x.strip()]
    if not groups:
        groups = list(GROUPS_DEFAULT)

    tag = _utc_now_tag()
    default_root = Path(__file__).resolve().parents[1] / "runs" / "acdc_batch_regression" / tag
    out_root = Path(args.runs_root).expanduser().resolve() if args.runs_root else default_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    acdc_root = Path(args.acdc_root).expanduser().resolve()
    grouped = _find_training_patients(acdc_root, str(args.split), groups)
    sampled = _sample_group_cases(grouped, int(args.n_per_group), int(args.seed), bool(args.random_sample))

    selected: List[Dict[str, Any]] = []
    for grp in groups:
        for p in sampled.get(grp, []):
            selected.append({"group": grp, "patient_id": p.name, "path": str(p)})
    (out_root / "selected_cases.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    if bool(args.dry_run):
        summary = _summarize(rows, groups)
        payload = {
            "args": vars(args),
            "out_root": str(out_root),
            "selected_cases": selected,
            "summary": summary,
        }
        (out_root / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[DRY-RUN] Selected {len(selected)} cases. Output: {out_root}")
        return

    agent_runs_root = out_root / "agent_runs"
    toolchain_root = out_root / "toolchain_runs"
    agent_runs_root.mkdir(parents=True, exist_ok=True)
    toolchain_root.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(selected, start=1):
        grp = str(item["group"])
        patient_id = str(item["patient_id"])
        patient_dir = Path(str(item["path"])).expanduser().resolve()
        case_id = f"{args.case_prefix}_{patient_id}_{idx:03d}"
        row: Dict[str, Any] = {
            "group": grp,
            "patient_id": patient_id,
            "mode": str(args.mode),
            "case_id": case_id,
        }
        try:
            if args.mode == "agent":
                out = _run_agent_case(
                    python_exe=str(args.python_exe),
                    case_id=case_id,
                    patient_dir=patient_dir,
                    case_runs_root=agent_runs_root,
                    llm_mode=str(args.llm_mode),
                    server_base_url=str(args.server_base_url),
                    server_model=str(args.server_model),
                    api_model=str(args.api_model),
                    api_base_url=str(args.api_base_url),
                    max_steps=int(args.max_steps),
                    autofix_mode=str(args.autofix_mode),
                )
            else:
                out = _run_toolchain_case(
                    case_id=case_id,
                    patient_dir=patient_dir,
                    run_root=toolchain_root,
                    cmr_reverse_root=Path(args.cmr_reverse_root).expanduser().resolve(),
                    nnunet_python=str(args.nnunet_python),
                    results_folder=str(args.results_folder),
                    with_features=bool(args.with_features),
                )
            row.update(out)
            row.update(_extract_metrics_metrics(out))
        except Exception as e:
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                    "predicted_group": None,
                    "ground_truth_group": None,
                    "ground_truth_match": None,
                }
            )
        rows.append(row)
        print(f"[{idx}/{len(selected)}] {patient_id} ({grp}) -> {row.get('status')} pred={row.get('predicted_group')}")

    summary = _summarize(rows, groups)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "out_root": str(out_root),
        "selected_cases": selected,
        "summary": summary,
        "rows": rows,
    }
    summary_json = out_root / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(rows, out_root / "results.csv")
    print(f"[DONE] Summary: {summary_json}")
    print(f"[DONE] Accuracy: {summary.get('accuracy')}")


if __name__ == "__main__":
    main()
