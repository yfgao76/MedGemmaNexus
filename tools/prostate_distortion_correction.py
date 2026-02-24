"""
Tool: correct_prostate_distortion

Purpose:
- Run the external prostate distortion correction pipeline (T2 + CNN + diffusion)
  from Prostate_distortion_recover as an agent tool.
- This tool is intended for prostate domain quality recovery workflows.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..commands.registry import Tool
from ..commands.schemas import ArtifactRef, ToolContext, ToolSpec
from ..core.paths import project_root


SPEC = ToolSpec(
    name="correct_prostate_distortion",
    description=(
        "Run prostate distortion correction using an external T2+CNN+diffusion inference pipeline. "
        "Wraps scripts/infer_diffusion_diffusers.py from Prostate_distortion_recover."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "python_exec": {"type": "string"},
            "project_root": {"type": "string"},
            "script_path": {"type": "string"},
            "ckpt": {"type": "string"},
            "cnn_ckpt": {"type": "string"},
            "test_root": {"type": "array"},
            "out_dir": {"type": "string"},
            "output_subdir": {"type": "string"},
            "device": {"type": "string"},
            "radius": {"type": "integer"},
            "max_cases": {"type": "integer"},
            "steps": {"type": "integer"},
            "strength": {"type": "number"},
            "eta": {"type": "number"},
            "sampler": {"type": "string"},
            "b_low": {"type": "number"},
            "b_high": {"type": "number"},
            "slice_mode": {"type": "string"},
            "save_npz": {"type": "boolean"},
            "save_slices": {"type": "boolean"},
            "gamma_b14": {"type": "number"},
            "gamma_adc": {"type": "number"},
            "adc_plo": {"type": "number"},
            "adc_phi": {"type": "number"},
            "t2_cond_channels": {"type": "integer"},
            "t2_contrast_mod": {"type": "string"},
            "t2_canny_low": {"type": "integer"},
            "t2_canny_high": {"type": "integer"},
            "cnn_base_channels": {"type": "integer"},
            "cnn_latent_dim": {"type": "integer"},
            "cnn_prompt_k": {"type": "integer"},
            "cnn_prompt_temp": {"type": "number"},
            "num_gpus": {"type": "integer"},
            "timeout_sec": {"type": "integer"},
        },
        "required": [],
    },
    output_schema={
        "type": "object",
        "properties": {
            "out_dir": {"type": "string"},
            "summary_json_path": {"type": "string"},
            "command": {"type": "string"},
            "num_pred_npz": {"type": "integer"},
            "num_panel_png": {"type": "integer"},
            "num_slice_png": {"type": "integer"},
            "runtime_sec": {"type": "number"},
            "stdout_tail": {"type": "string"},
            "stderr_tail": {"type": "string"},
            "note": {"type": "string"},
        },
        "required": ["out_dir", "summary_json_path"],
    },
    version="0.1.0",
    tags=["prostate", "distortion", "diffusion", "cnn"],
)


def _default_repo_root() -> Path:
    env = str(os.getenv("MRI_AGENT_PROSTATE_DISTORTION_ROOT") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
    candidates = [
        Path("/home/longz2/common/Prostate_distortion_recover"),
        project_root().parent / "Prostate_distortion_recover",
        project_root() / "external" / "Prostate_distortion_recover",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return candidates[0].resolve()


def _default_ckpt(repo_root: Path) -> Path:
    env = str(os.getenv("MRI_AGENT_PROSTATE_DISTORTION_DIFF_CKPT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (
        repo_root
        / "network_runs"
        / "diffusion_clean_v2"
        / "diff_t2cnn_clean_epoch_092.pt"
    ).resolve()


def _default_cnn_ckpt(repo_root: Path) -> Path:
    env = str(os.getenv("MRI_AGENT_PROSTATE_DISTORTION_CNN_CKPT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (
        repo_root
        / "network_runs"
        / "mageultra_dualb_axis01_quick_local_v25"
        / "mageultra_epoch_025.pt"
    ).resolve()


def _default_test_roots(repo_root: Path) -> List[str]:
    env = str(os.getenv("MRI_AGENT_PROSTATE_DISTORTION_TEST_ROOTS") or "").strip()
    if env:
        seps = [os.pathsep, ","]
        items = [env]
        for sep in seps:
            nxt: List[str] = []
            for it in items:
                nxt.extend(it.split(sep))
            items = nxt
        vals = [str(Path(x.strip()).expanduser().resolve()) for x in items if x.strip()]
        if vals:
            return vals
    return [
        str((repo_root / "validation_data2_TSE").resolve()),
        str((repo_root / "preprocessed_infer_npz").resolve()),
    ]


def _as_int(args: Dict[str, Any], key: str, default: int) -> int:
    try:
        if key in args and args[key] is not None:
            return int(args[key])
    except Exception:
        pass
    return int(default)


def _as_float(args: Dict[str, Any], key: str, default: float) -> float:
    try:
        if key in args and args[key] is not None:
            return float(args[key])
    except Exception:
        pass
    return float(default)


def _as_str(args: Dict[str, Any], key: str, default: str) -> str:
    v = args.get(key)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _as_bool(args: Dict[str, Any], key: str, default: bool) -> bool:
    if key not in args:
        return bool(default)
    v = args.get(key)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _tail_text(text: str, max_lines: int = 120) -> str:
    lines = str(text or "").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _resolve_test_roots(args: Dict[str, Any], repo_root: Path) -> List[str]:
    test_root = args.get("test_root")
    vals: List[str] = []
    if isinstance(test_root, (list, tuple)):
        for x in test_root:
            s = str(x).strip()
            if s:
                vals.append(str(Path(s).expanduser().resolve()))
    elif isinstance(test_root, str) and test_root.strip():
        vals.append(str(Path(test_root.strip()).expanduser().resolve()))
    if not vals:
        vals = _default_test_roots(repo_root)
    return vals


def correct_prostate_distortion(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    repo_root = Path(str(args.get("project_root") or _default_repo_root())).expanduser().resolve()
    script_path = Path(
        str(args.get("script_path") or (repo_root / "scripts" / "infer_diffusion_diffusers.py"))
    ).expanduser().resolve()
    ckpt = Path(str(args.get("ckpt") or _default_ckpt(repo_root))).expanduser().resolve()
    cnn_ckpt = Path(str(args.get("cnn_ckpt") or _default_cnn_ckpt(repo_root))).expanduser().resolve()

    out_dir_raw = str(args.get("out_dir") or "").strip()
    output_subdir = _as_str(args, "output_subdir", "distortion_correction")
    if out_dir_raw:
        out_dir = Path(out_dir_raw).expanduser().resolve()
    else:
        out_dir = (ctx.artifacts_dir / output_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    test_roots = _resolve_test_roots(args, repo_root)
    missing_test_roots = [p for p in test_roots if not Path(p).exists()]
    warnings: List[str] = []
    if missing_test_roots:
        warnings.append("Some test_root paths do not exist: " + ", ".join(missing_test_roots))

    if not repo_root.exists():
        raise RuntimeError(f"project_root not found: {repo_root}")
    if not script_path.exists():
        raise RuntimeError(f"inference script not found: {script_path}")
    if not ckpt.exists():
        raise RuntimeError(f"diffusion checkpoint not found: {ckpt}")
    if not cnn_ckpt.exists():
        raise RuntimeError(f"CNN checkpoint not found: {cnn_ckpt}")

    cmd: List[str] = [
        _as_str(args, "python_exec", "python"),
        "-u",
        str(script_path),
        "--ckpt",
        str(ckpt),
        "--cnn_ckpt",
        str(cnn_ckpt),
        "--test_root",
        *test_roots,
        "--out_dir",
        str(out_dir),
        "--radius",
        str(_as_int(args, "radius", 2)),
        "--steps",
        str(_as_int(args, "steps", 100)),
        "--strength",
        str(_as_float(args, "strength", 0.1)),
        "--eta",
        str(_as_float(args, "eta", 0.0)),
        "--sampler",
        _as_str(args, "sampler", "dpmsolver"),
        "--slice_mode",
        _as_str(args, "slice_mode", "all"),
        "--b_low",
        str(_as_float(args, "b_low", 50.0)),
        "--b_high",
        str(_as_float(args, "b_high", 1400.0)),
        "--gamma_b14",
        str(_as_float(args, "gamma_b14", 0.4)),
        "--gamma_adc",
        str(_as_float(args, "gamma_adc", 0.7)),
        "--adc_plo",
        str(_as_float(args, "adc_plo", 2.0)),
        "--adc_phi",
        str(_as_float(args, "adc_phi", 99.0)),
        "--t2_cond_channels",
        str(_as_int(args, "t2_cond_channels", 64)),
        "--t2_contrast_mod",
        _as_str(args, "t2_contrast_mod", "none"),
        "--t2_canny_low",
        str(_as_int(args, "t2_canny_low", 50)),
        "--t2_canny_high",
        str(_as_int(args, "t2_canny_high", 150)),
        "--cnn_base_channels",
        str(_as_int(args, "cnn_base_channels", 64)),
        "--cnn_latent_dim",
        str(_as_int(args, "cnn_latent_dim", 8)),
        "--cnn_prompt_k",
        str(_as_int(args, "cnn_prompt_k", 8)),
        "--cnn_prompt_temp",
        str(_as_float(args, "cnn_prompt_temp", 1.0)),
    ]

    device = str(args.get("device") or "").strip()
    if device:
        cmd.extend(["--device", device])
    num_gpus = args.get("num_gpus")
    if num_gpus is not None:
        cmd.extend(["--num_gpus", str(int(num_gpus))])
    max_cases = args.get("max_cases")
    if max_cases is not None:
        cmd.extend(["--max_cases", str(int(max_cases))])
    if _as_bool(args, "save_npz", True):
        cmd.append("--save_npz")
    if _as_bool(args, "save_slices", True):
        cmd.append("--save_slices")

    timeout_sec = _as_int(args, "timeout_sec", 6 * 60 * 60)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    runtime_sec = float(time.time() - t0)

    stdout_tail = _tail_text(proc.stdout, max_lines=120)
    stderr_tail = _tail_text(proc.stderr, max_lines=120)
    if proc.returncode != 0:
        raise RuntimeError(
            "correct_prostate_distortion failed "
            f"(exit={proc.returncode}).\n"
            f"STDOUT tail:\n{stdout_tail}\n\nSTDERR tail:\n{stderr_tail}"
        )

    pred_npz = sorted(out_dir.glob("*_pred.npz"))
    panel_png = sorted(out_dir.glob("*_panel.png"))
    slice_png = sorted(out_dir.glob("*_slices/*.png"))
    summary_path = out_dir / "distortion_correction_summary.json"
    summary = {
        "tool": SPEC.name,
        "version": SPEC.version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(repo_root),
        "script_path": str(script_path),
        "ckpt": str(ckpt),
        "cnn_ckpt": str(cnn_ckpt),
        "test_root": test_roots,
        "out_dir": str(out_dir),
        "command": shlex.join(cmd),
        "runtime_sec": runtime_sec,
        "return_code": int(proc.returncode),
        "num_pred_npz": len(pred_npz),
        "num_panel_png": len(panel_png),
        "num_slice_png": len(slice_png),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "warnings": warnings,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    artifacts: List[ArtifactRef] = [
        ArtifactRef(path=str(summary_path), kind="json", description="Distortion correction run summary"),
        ArtifactRef(path=str(out_dir), kind="directory", description="Distortion correction output directory"),
    ]
    for p in panel_png[:8]:
        artifacts.append(ArtifactRef(path=str(p), kind="figure", description="Distortion correction panel"))
    for p in pred_npz[:8]:
        artifacts.append(ArtifactRef(path=str(p), kind="npz", description="Distortion correction prediction NPZ"))

    source_artifacts: List[ArtifactRef] = [
        ArtifactRef(path=str(ckpt), kind="checkpoint", description="Diffusion checkpoint"),
        ArtifactRef(path=str(cnn_ckpt), kind="checkpoint", description="CNN checkpoint"),
    ]
    for tr in test_roots[:6]:
        source_artifacts.append(ArtifactRef(path=str(tr), kind="directory", description="Inference input root"))

    return {
        "data": {
            "out_dir": str(out_dir),
            "summary_json_path": str(summary_path),
            "command": shlex.join(cmd),
            "num_pred_npz": len(pred_npz),
            "num_panel_png": len(panel_png),
            "num_slice_png": len(slice_png),
            "runtime_sec": runtime_sec,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "note": "T2+CNN diffusion distortion correction completed.",
        },
        "artifacts": artifacts,
        "warnings": warnings,
        "source_artifacts": source_artifacts,
        "generated_artifacts": artifacts,
    }


def build_tool() -> Tool:
    return Tool(spec=SPEC, func=correct_prostate_distortion)
