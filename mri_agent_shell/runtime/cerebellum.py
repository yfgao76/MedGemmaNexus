from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from jsonschema import ValidationError, validate

from ..bootstrap import ensure_import_paths
from ..io.report_writer import ensure_report
from ..runtime.events import append_event, event_record, summarize_args, summarize_outputs
from ..runtime.session import SessionState

ensure_import_paths()

from MRI_Agent.commands.dispatcher import ToolDispatcher, validate_args_minimal  # noqa: E402
from MRI_Agent.commands.schemas import CaseState, ToolCall, ToolContext  # noqa: E402
from MRI_Agent.core.domain_config import get_domain_config  # noqa: E402
from MRI_Agent.core.paths import project_root  # noqa: E402
from MRI_Agent.core.plan_dag import AgentPlanDAG, PlanNode, legacy_plan_to_dag  # noqa: E402
from MRI_Agent.tools.arg_models import RepairConfig, repair_tool_args  # noqa: E402


class ScopeViolation(RuntimeError):
    pass


def _safe_resolve(path_like: str) -> Path:
    return Path(str(path_like)).expanduser().resolve()


def _safe_abspath(path_like: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path_like))))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _guess_case_ref_from_session(session: SessionState) -> str:
    return str(session.case_inputs.get("case_input_path") or session.path_keys.get("case.input") or "").strip()


def _default_external_model_roots() -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()

    def _push(path_like: Any) -> None:
        raw = str(path_like or "").strip()
        if not raw:
            return
        try:
            p = _safe_resolve(raw)
        except Exception:
            return
        key = str(p)
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    _push(Path("~/.cache/torch/hub/bundle").expanduser())
    _push(Path("~/.cache/torch/hub").expanduser())

    try:
        cfg = RepairConfig.from_env(project_root())
        _push(cfg.model_registry_path)
        _push(cfg.prostate_bundle_dir)
        _push(cfg.lesion_weights_dir)
        _push(cfg.brain_bundle_dir)
        _push(cfg.cardiac_cmr_reverse_root)
        _push(cfg.cardiac_results_folder)
        _push(cfg.cardiac_nnunet_python)
        _push(Path(cfg.cardiac_nnunet_python).expanduser().resolve().parent)
        _push(cfg.distortion_repo_root)
        _push(cfg.distortion_diff_ckpt)
        _push(cfg.distortion_cnn_ckpt)
        for p in (cfg.distortion_test_roots or []):
            _push(p)
    except Exception:
        pass
    return out


@dataclass
class _ScopedBinder:
    case_root: Path
    refs: Dict[str, str] = field(default_factory=dict)
    seq_paths: Dict[str, str] = field(default_factory=dict)
    node_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def set_ref(self, key: str, value: str) -> None:
        k = str(key or "").strip()
        v = str(value or "").strip()
        if not k or not v:
            return
        self.refs[k] = v

    def seed_runtime(self, ctx: ToolContext) -> None:
        self.set_ref("runtime.run_dir", str(ctx.run_dir))
        self.set_ref("runtime.artifacts_dir", str(ctx.artifacts_dir))
        self.set_ref("runtime.case_state_path", str(ctx.case_state_path))
        self.set_ref("case.input", str(self.case_root))

    def resolve_refs(self, obj: Any) -> Any:
        token_aliases = {
            "state.path": "runtime.case_state_path",
            "case.path": "case.input",
        }

        def _resolve(v: Any) -> Any:
            if isinstance(v, dict):
                return {k: _resolve(x) for k, x in v.items()}
            if isinstance(v, list):
                return [_resolve(x) for x in v]
            if isinstance(v, str) and v.startswith("@"):
                token = v[1:]
                if token.startswith("node."):
                    parts = token.split(".")
                    if len(parts) >= 3:
                        node_id = parts[1]
                        key_path = parts[2:]
                        cur: Any = self.node_outputs.get(node_id)
                        for seg in key_path:
                            if isinstance(cur, dict) and seg in cur:
                                cur = cur.get(seg)
                            else:
                                cur = None
                                break
                        if cur is not None:
                            return cur
                mapped = self.refs.get(token)
                if mapped is not None and str(mapped).strip():
                    return mapped
                alias = token_aliases.get(token)
                if alias:
                    alias_val = self.refs.get(alias)
                    if alias_val is not None and str(alias_val).strip():
                        return alias_val
                return v
            return v

        return _resolve(obj)

    def _normalize_candidate_path(self, p: Path, *, prefer_file: bool) -> str:
        try:
            pp = p.expanduser()
            if pp.exists() and pp.is_file():
                return str(_safe_abspath(str(pp)))
            if pp.exists() and pp.is_dir() and prefer_file:
                picked = self._pick_nifti_from_dir(pp)
                if picked is not None:
                    return str(_safe_abspath(str(picked)))
            if pp.exists() and pp.is_dir():
                return str(_safe_abspath(str(pp)))
            return str(pp)
        except Exception:
            return str(p)

    def _pick_nifti_from_dir(self, path: Path) -> Optional[Path]:
        try:
            if not path.exists() or not path.is_dir():
                return None
            direct = sorted([x for x in path.iterdir() if x.is_file() and x.name.lower().endswith((".nii", ".nii.gz"))])
            if direct:
                return direct[0]
            deep = sorted([x for x in path.rglob("*") if x.is_file() and x.name.lower().endswith((".nii", ".nii.gz"))])
            if deep:
                return deep[0]
        except Exception:
            return None
        return None

    def resolve_sequence_or_case_path(
        self,
        raw: str,
        *,
        preferred_tokens: Optional[List[str]] = None,
        prefer_file: bool = False,
    ) -> Optional[str]:
        s = str(raw or "").strip()
        if not s or s.startswith("@"):
            return None

        try:
            p = Path(s).expanduser()
            if p.is_absolute():
                return self._normalize_candidate_path(p, prefer_file=prefer_file)
        except Exception:
            pass

        aliases = {
            "t2": "T2w",
            "t2w": "T2w",
            "t1c": "T1c",
            "t1": "T1",
            "flair": "FLAIR",
            "adc": "ADC",
            "dwi": "DWI",
            "cine": "CINE",
        }

        # Always honor the explicit node token first (e.g., DWI), then alias,
        # and only then fallback to preferred tokens.
        candidates: List[str] = [s]
        alias = aliases.get(s.lower(), s)
        if str(alias).strip() and str(alias) != s:
            candidates.append(str(alias))
        if preferred_tokens:
            candidates.extend(preferred_tokens)

        seen = set()
        for name in candidates:
            n = str(name or "").strip()
            if not n or n in seen:
                continue
            seen.add(n)
            v = str(self.seq_paths.get(n) or "").strip()
            if not v:
                continue
            try:
                p = Path(v).expanduser()
                if p.exists() or p.is_absolute():
                    resolved = self._normalize_candidate_path(p, prefer_file=prefer_file)
                    if resolved:
                        return resolved
            except Exception:
                return v

        case_dir = self.case_root
        direct_candidates = [
            case_dir / s,
            case_dir / f"{s}.nii.gz",
            case_dir / f"{s}.nii",
            case_dir / s.lower(),
            case_dir / f"{s.lower()}.nii.gz",
            case_dir / f"{s.lower()}.nii",
        ]
        for cand in direct_candidates:
            if cand.exists():
                resolved = self._normalize_candidate_path(cand, prefer_file=prefer_file)
                if resolved:
                    return resolved

        low = s.lower()
        hint_map = {
            "t2w": ("t2", "t2w"),
            "t2": ("t2", "t2w"),
            "adc": ("adc",),
            "dwi": ("dwi", "diff", "trace", "high_b", "low_b"),
            "t1c": ("t1c", "t1ce", "t1gd"),
            "t1": ("t1",),
            "flair": ("flair",),
            "cine": ("cine", "bssfp", "ssfp", "sax"),
        }
        hints = hint_map.get(low, (low,))
        try:
            for p in sorted(case_dir.rglob("*")):
                n = p.name.lower()
                if not any(h in n for h in hints):
                    continue
                if p.is_file() and n.endswith((".nii", ".nii.gz", ".dcm")):
                    return str(_safe_abspath(str(p)))
                if p.is_dir():
                    resolved = self._normalize_candidate_path(p, prefer_file=prefer_file)
                    if resolved:
                        return resolved
        except Exception:
            return None
        return None

    def learn_from_tool(self, *, tool_name: str, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return

        for k, v in data.items():
            if isinstance(v, str) and k.endswith("_path"):
                self.set_ref(f"{tool_name}.{k[:-5]}", v)

        if tool_name == "identify_sequences":
            mapping = data.get("mapping") if isinstance(data.get("mapping"), dict) else {}
            for name, path in mapping.items():
                if isinstance(path, str) and path:
                    self.seq_paths[str(name)] = str(path)
                    self.set_ref(f"seq.{name}", str(path))
            t2w_ref = str(mapping.get("T2w") or "").strip()
            if not t2w_ref:
                t2_fallback = ""
                if isinstance(mapping.get("T2"), str) and str(mapping.get("T2")).strip():
                    t2_fallback = str(mapping.get("T2")).strip()
                else:
                    for key, path in mapping.items():
                        if not isinstance(path, str) or not str(path).strip():
                            continue
                        key_l = str(key).strip().lower()
                        if key_l in {"t2", "t2w"} or "t2w" in key_l or key_l.startswith("t2"):
                            t2_fallback = str(path).strip()
                            break
                if t2_fallback:
                    self.seq_paths["T2w"] = t2_fallback
                    self.set_ref("seq.T2w", t2_fallback)
            # Some identify mappings only include ADC/DWI. Recover T2w from case structure for downstream refs.
            if not str(self.seq_paths.get("T2w") or "").strip():
                discovered_t2 = self.resolve_sequence_or_case_path(
                    "t2w",
                    preferred_tokens=["T2w", "T2"],
                    prefer_file=True,
                )
                if discovered_t2:
                    self.seq_paths["T2w"] = str(discovered_t2)
                    self.set_ref("seq.T2w", str(discovered_t2))
            if isinstance(mapping.get("T1"), str) and not str(mapping.get("T1c") or "").strip():
                self.seq_paths["T1c"] = str(mapping.get("T1"))
                self.set_ref("seq.T1c", str(mapping.get("T1")))

    def learn_from_node(self, *, node_id: str, data: Dict[str, Any]) -> None:
        self.node_outputs[str(node_id)] = dict(data or {})


@dataclass
class _CaseScopeGuard:
    case_root: Path
    run_root: Path
    external_roots: List[Path] = field(default_factory=list)
    _case_symlink_target_roots: Optional[List[Path]] = field(default=None, init=False, repr=False)

    _EXTERNAL_ARG_KEYS = {
        "bundle_dir",
        "bundle_root",
        "weights_dir",
        "nnunet_python",
        "results_folder",
        "cmr_reverse_root",
        "project_root",
        "repo_root",
        "test_root",
        "python_exec",
        "ckpt",
        "cnn_ckpt",
        "checkpoint",
        "model_registry",
        "model_registry_path",
        "server_base_url",
        "api_base_url",
    }

    _NON_PATH_ARG_KEYS = {
        "output_subdir",
        "method",
        "domain",
        "llm_mode",
        "api_model",
        "temperature",
        "max_tokens",
        "device",
        "note",
    }

    _MUST_BE_ABS_KEYS = {
        "dicom_case_dir",
        "series_inventory_path",
        "case_state_path",
        "case_path",
        "fixed",
        "moving",
        "t2w_ref",
        "t1c_path",
        "t1_path",
        "t2_path",
        "flair_path",
        "cine_path",
        "seg_path",
        "ed_seg_path",
        "es_seg_path",
        "patient_info_path",
    }

    def _is_path_candidate(self, key: str, value: str) -> bool:
        k = str(key or "").strip().lower()
        v = str(value or "").strip()
        if not v or k in self._NON_PATH_ARG_KEYS:
            return False
        if k in self._EXTERNAL_ARG_KEYS:
            return True
        if k in self._MUST_BE_ABS_KEYS:
            return True
        if k.endswith("_path") or k.endswith("_dir") or k.endswith("_root") or k.endswith("_ckpt"):
            if k == "output_subdir":
                return False
            return True
        if v.startswith(("/", "~", "./", "../")):
            return True
        if v.lower().endswith((
            ".nii",
            ".nii.gz",
            ".dcm",
            ".json",
            ".txt",
            ".md",
            ".png",
            ".jpg",
            ".jpeg",
            ".tfm",
            ".pt",
            ".pth",
            ".ckpt",
        )):
            return True
        return False

    def _walk_path_args(self, obj: Any, key_hint: str = "") -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.extend(self._walk_path_args(v, key_hint=str(k)))
            return out
        if isinstance(obj, list):
            for v in obj:
                out.extend(self._walk_path_args(v, key_hint=key_hint))
            return out
        if isinstance(obj, str) and self._is_path_candidate(key_hint, obj):
            out.append((key_hint, obj))
        return out

    def _allow_external(self, resolved: Path) -> bool:
        for root in self.external_roots:
            if _is_relative_to(resolved, root):
                return True
        return False

    def _collect_case_symlink_target_roots(self) -> List[Path]:
        cached = self._case_symlink_target_roots
        if isinstance(cached, list):
            return cached
        out: List[Path] = []
        seen: set[str] = set()

        def _push(p: Path) -> None:
            try:
                rp = p.resolve()
            except Exception:
                return
            key = str(rp)
            if key in seen:
                return
            seen.add(key)
            out.append(rp)

        try:
            for p in self.case_root.rglob("*"):
                if not p.is_symlink():
                    continue
                try:
                    target = p.resolve()
                except Exception:
                    continue
                if target.is_dir():
                    _push(target)
                else:
                    _push(target.parent)
        except Exception:
            pass
        self._case_symlink_target_roots = out
        return out

    def _allow_case_symlink_target(self, resolved: Path) -> bool:
        for root in self._collect_case_symlink_target_roots():
            if _is_relative_to(resolved, root):
                return True
        return False

    def validate_args(self, *, tool_name: str, args: Dict[str, Any]) -> None:
        for key, raw in self._walk_path_args(args):
            value = str(raw or "").strip()
            key_l = str(key or "").strip().lower()
            if not value:
                continue
            if value.startswith("@"):
                raise ScopeViolation(f"unresolved reference in path argument '{key}' for {tool_name}: {value}")

            if key_l in self._EXTERNAL_ARG_KEYS:
                try:
                    p = Path(value).expanduser()
                    if p.is_absolute() and not self._allow_external(p.resolve()):
                        # External keys are allowed, but if a root allowlist is provided, enforce it.
                        if self.external_roots:
                            raise ScopeViolation(
                                f"external path '{p}' for '{key}' is outside allow_external_model_roots"
                            )
                except ScopeViolation:
                    raise
                except Exception:
                    pass
                continue

            p = Path(value).expanduser()
            if key_l in self._MUST_BE_ABS_KEYS and (not p.is_absolute()):
                raise ScopeViolation(f"path argument '{key}' must be absolute for {tool_name}: {value}")

            if not p.is_absolute():
                # Relative path-like values are allowed only if they are clearly runtime-local outputs.
                continue

            # Symlink-aware lexical scope check: if the user-facing path itself is
            # inside case/run scope, allow it even if the symlink target resolves
            # outside (common for shared dataset links).
            try:
                lexical = Path(os.path.abspath(str(p)))
            except Exception:
                lexical = p
            if _is_relative_to(lexical, self.case_root):
                continue
            if _is_relative_to(lexical, self.run_root):
                continue

            resolved = p.resolve()
            if _is_relative_to(resolved, self.case_root):
                continue
            if _is_relative_to(resolved, self.run_root):
                continue
            if self._allow_case_symlink_target(resolved):
                continue
            raise ScopeViolation(
                f"path argument '{key}' escaped case scope for {tool_name}: {resolved}"
            )


class Cerebellum:
    def __init__(
        self,
        *,
        session: SessionState,
        registry: Any,
        max_attempts: int = 2,
        reflect_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        failure_reflect_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.session = session
        self.registry = registry
        self.max_attempts = max(1, int(max_attempts))
        self._last_scope_key = ""
        self._reflect_fn = reflect_fn or self._default_reflect_fn
        self._failure_reflect_fn = failure_reflect_fn or self._default_failure_reflect_fn
        self._failure_retry_limit = 1
        self._reflection_llm: Any = None
        self._reflection_llm_ready = False

        self.dispatcher = ToolDispatcher(
            registry=self.registry,
            runs_root=Path(self.session.runs_root).expanduser().resolve(),
        )

    def _default_reflect_fn(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        decision = str(args.get("decision") or "continue").strip().lower() if isinstance(args, dict) else "continue"
        if decision not in {"continue", "halt"}:
            decision = "continue"
        reason = str(args.get("reason") or "no-op control node") if isinstance(args, dict) else "no-op control node"
        return {"decision": decision, "reason": reason}

    def _build_reflection_llm(self) -> Any:
        if self._reflection_llm_ready:
            return self._reflection_llm
        self._reflection_llm_ready = True
        cfg = self.session.model_config
        provider = str(getattr(cfg, "provider", "") or "").strip().lower()
        try:
            if provider == "openai_compatible_server":
                from MRI_Agent.llm.adapter_vllm_server import VLLMOpenAIChatAdapter, VLLMServerConfig  # noqa: E402

                self._reflection_llm = VLLMOpenAIChatAdapter(
                    VLLMServerConfig(
                        base_url=str(getattr(cfg, "server_base_url", "") or "http://127.0.0.1:8000/v1"),
                        model=str(getattr(cfg, "llm", "") or ""),
                        api_key=str(getattr(cfg, "effective_api_key", lambda: "")() or "EMPTY"),
                        max_tokens=int(min(512, int(getattr(cfg, "max_tokens", 512) or 512))),
                        temperature=float(getattr(cfg, "temperature", 0.0) or 0.0),
                    )
                )
                return self._reflection_llm
            if provider == "openai_official":
                from MRI_Agent.llm.adapter_openai_api import OpenAIChatAdapter, OpenAIConfig  # noqa: E402

                self._reflection_llm = OpenAIChatAdapter(
                    OpenAIConfig(
                        model=str(getattr(cfg, "llm", "") or ""),
                        api_key=str(getattr(cfg, "effective_api_key", lambda: "")() or ""),
                        base_url=(str(getattr(cfg, "api_base_url", "") or "").strip() or None),
                        max_tokens=int(min(512, int(getattr(cfg, "max_tokens", 512) or 512))),
                        temperature=float(getattr(cfg, "temperature", 0.0) or 0.0),
                    )
                )
                return self._reflection_llm
            if provider == "anthropic":
                from MRI_Agent.llm.adapter_anthropic_api import AnthropicChatAdapter, AnthropicConfig  # noqa: E402

                self._reflection_llm = AnthropicChatAdapter(
                    AnthropicConfig(
                        model=str(getattr(cfg, "llm", "") or ""),
                        api_key=str(getattr(cfg, "effective_api_key", lambda: "")() or ""),
                        max_tokens=int(min(512, int(getattr(cfg, "max_tokens", 512) or 512))),
                        temperature=float(getattr(cfg, "temperature", 0.0) or 0.0),
                    )
                )
                return self._reflection_llm
            if provider == "gemini":
                from MRI_Agent.llm.adapter_gemini_api import GeminiChatAdapter, GeminiConfig  # noqa: E402

                self._reflection_llm = GeminiChatAdapter(
                    GeminiConfig(
                        model=str(getattr(cfg, "llm", "") or ""),
                        api_key=str(getattr(cfg, "effective_api_key", lambda: "")() or ""),
                        max_tokens=int(min(512, int(getattr(cfg, "max_tokens", 512) or 512))),
                        temperature=float(getattr(cfg, "temperature", 0.0) or 0.0),
                    )
                )
                return self._reflection_llm
        except Exception:
            self._reflection_llm = None
            return None
        self._reflection_llm = None
        return None

    def _parse_reflection_json(self, text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                obj = json.loads(str(m.group(0)))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return {}

    def _collect_reflection_binding_context(self, *, binder: _ScopedBinder) -> Dict[str, Any]:
        token_bindings: Dict[str, str] = {}
        for k, v in (binder.refs or {}).items():
            kk = str(k or "").strip()
            vv = str(v or "").strip()
            if kk and vv:
                token_bindings[f"@{kk}"] = vv

        session_token_bindings: Dict[str, str] = {}
        for k, v in (self.session.path_keys or {}).items():
            kk = str(k or "").strip()
            vv = str(v or "").strip()
            if kk and vv:
                session_token_bindings[f"@{kk}"] = vv

        return {
            "available_valid_tokens": sorted(token_bindings.keys()),
            "token_bindings": token_bindings,
            "session_path_keys": sorted(session_token_bindings.keys()),
            "session_token_bindings": session_token_bindings,
            "available_modalities": sorted([str(k) for k in (binder.seq_paths or {}).keys() if str(k).strip()]),
        }

    def _apply_ref_token_aliases(self, obj: Any) -> Tuple[Any, bool]:
        aliases = {
            "@state.path": "@runtime.case_state_path",
            "state.path": "@runtime.case_state_path",
            "@case.path": "@case.input",
            "case.path": "@case.input",
        }
        changed = False

        def _map(v: Any) -> Any:
            nonlocal changed
            if isinstance(v, dict):
                return {k: _map(x) for k, x in v.items()}
            if isinstance(v, list):
                return [_map(x) for x in v]
            if isinstance(v, str) and v in aliases:
                changed = True
                return aliases[v]
            return v

        return _map(obj), changed

    def _deterministic_retry_suggestion(
        self,
        *,
        node: PlanNode,
        last_args: Dict[str, Any],
        error: Dict[str, Any],
        binding_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        out = dict(last_args or {})
        changed = False

        out, alias_changed = self._apply_ref_token_aliases(out)
        changed = changed or alias_changed

        msg = str((error or {}).get("message") or "")
        m = re.search(r"unresolved reference in path argument '([^']+)'[^:]*: (\S+)", msg)
        unresolved_key = str(m.group(1) or "").strip().lower() if m else ""
        unresolved_ref = str(m.group(2) or "").strip() if m else ""

        valid_tokens = set(
            str(x or "").strip() for x in (binding_context.get("available_valid_tokens") or []) if str(x or "").strip()
        )

        def _can_use(tok: str) -> bool:
            if tok in valid_tokens:
                return True
            return tok in {"@runtime.case_state_path", "@case.input"}

        if unresolved_key in {"case_state_path", "state_path"} and _can_use("@runtime.case_state_path"):
            out[unresolved_key] = "@runtime.case_state_path"
            changed = True
        elif unresolved_key in {"case_path", "dicom_case_dir"} and _can_use("@case.input"):
            out[unresolved_key] = "@case.input"
            changed = True
        elif unresolved_ref in {"@state.path", "state.path"} and _can_use("@runtime.case_state_path"):
            if unresolved_key:
                out[unresolved_key] = "@runtime.case_state_path"
            changed = True

        if str(node.tool_name or "").strip() == "rag_search":
            raw = str(out.get("case_state_path") or "").strip()
            if (not raw or raw in {"@state.path", "state.path"}) and _can_use("@runtime.case_state_path"):
                out["case_state_path"] = "@runtime.case_state_path"
                changed = True

        return out if changed else {}

    def _classify_reflection_limit(self, *, err_type: str, err_msg: str, deterministic_retry: Dict[str, Any]) -> str:
        et = str(err_type or "").strip().lower()
        msg = str(err_msg or "").strip().lower()
        if deterministic_retry:
            return "fixable_runtime"
        if et == "schemavalidationerror" or "not allowed for domain" in msg:
            return "hard_limit_schema"
        if et == "scopeviolation":
            if "unresolved reference in path argument" in msg:
                return "fixable_runtime"
            return "hard_limit_scope"
        return "unknown_runtime"

    def _hard_limit_response(
        self,
        *,
        tool_name: str,
        err_type: str,
        err_msg: str,
        limit_class: str,
    ) -> Dict[str, Any]:
        if limit_class == "hard_limit_schema":
            nl = (
                f"I encountered a {err_type} for '{tool_name}': {err_msg}. "
                "As a runtime execution agent, I do not have permission to modify system schema configurations. "
                "Please have a developer add this tool to the domain whitelist or update the backend schema rules."
            )
            return {
                "action": "halt",
                "reason": f"hard_limit_schema: {err_type}: {err_msg}",
                "retry_arguments": {},
                "natural_language_response": nl,
            }
        if limit_class == "hard_limit_scope":
            nl = (
                f"I encountered a {err_type} for '{tool_name}': {err_msg}. "
                "As a runtime execution agent, I cannot bypass case scope guard restrictions. "
                "Please adjust case paths or policy configuration in code."
            )
            return {
                "action": "halt",
                "reason": f"hard_limit_scope: {err_type}: {err_msg}",
                "retry_arguments": {},
                "natural_language_response": nl,
            }
        return {
            "action": "halt",
            "reason": f"{err_type}: {err_msg}",
            "retry_arguments": {},
            "natural_language_response": (
                f"I cannot continue because required step '{tool_name}' failed: {err_msg}."
            ),
        }

    def _default_failure_reflect_fn(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        err = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        err_type = str((err or {}).get("type") or "RuntimeError")
        err_msg = str((err or {}).get("message") or "required node failed")
        tool_name = str(payload.get("tool_name") or "unknown_tool")
        deterministic_retry = payload.get("deterministic_retry_suggestion")
        deterministic_retry = (
            dict(deterministic_retry)
            if isinstance(deterministic_retry, dict) and deterministic_retry
            else {}
        )
        limit_class = self._classify_reflection_limit(
            err_type=err_type,
            err_msg=err_msg,
            deterministic_retry=deterministic_retry,
        )
        if limit_class in {"hard_limit_schema", "hard_limit_scope"}:
            return self._hard_limit_response(
                tool_name=tool_name,
                err_type=err_type,
                err_msg=err_msg,
                limit_class=limit_class,
            )
        fallback = {
            "action": "halt",
            "reason": f"{err_type}: {err_msg}",
            "retry_arguments": {},
            "natural_language_response": (
                f"I cannot continue because required step '{tool_name}' failed: {err_msg}."
            ),
        }
        if deterministic_retry:
            fallback = {
                "action": "retry",
                "reason": f"auto_fix_unresolved_path: {err_type}: {err_msg}",
                "retry_arguments": deterministic_retry,
                "natural_language_response": (
                    f"I corrected a path/token binding for required step '{tool_name}' and will retry."
                ),
            }
        llm = self._build_reflection_llm()
        if llm is None:
            return fallback
        prompt = {
            "instruction": (
                "Return JSON only with keys: action, reason, retry_arguments, natural_language_response. "
                "action must be one of halt|skip|retry. "
                "You are an active debugger for MRI DAG execution with strict boundaries. "
                "CAN: fix runtime argument/token/path mismatches, adjust safe hyperparameters, and retry with corrected arguments. "
                "CANNOT: bypass ScopeViolation policy boundaries, bypass SchemaValidationError domain/schema rules, or modify backend Python code/config files. "
                "For fixable runtime issues, choose action=retry with corrected retry_arguments. "
                "For hard limits, choose action=halt and explicitly explain why it is outside runtime authority."
            ),
            "reflector_boundaries": {
                "can": [
                    "fix runtime token/path argument mismatches",
                    "adjust runtime hyperparameters and retry",
                    "swap to valid in-scope paths/tokens from binding context",
                ],
                "cannot": [
                    "modify domain/schema whitelist",
                    "bypass case scope policy",
                    "edit backend source code",
                ],
            },
            "failure_classification": limit_class,
            "failure": payload,
        }
        msgs = [
            {
                "role": "system",
                "content": (
                    "You are an active MRI execution debugger with bounded authority. Output strict JSON only. "
                    "Use retry only for fixable runtime arguments/path/token issues. "
                    "For hard limits (schema/scope/backend-policy), halt and explicitly state that runtime cannot change backend configuration or code."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        try:
            raw = str(llm.generate(msgs)).strip()
            obj = self._parse_reflection_json(raw)
            action = str(obj.get("action") or "").strip().lower()
            if action not in {"halt", "skip", "retry"}:
                return fallback
            out = dict(fallback)
            out["action"] = action
            out["reason"] = str(obj.get("reason") or fallback["reason"]).strip()
            out["natural_language_response"] = str(
                obj.get("natural_language_response") or fallback["natural_language_response"]
            ).strip()
            retry_args = obj.get("retry_arguments")
            out["retry_arguments"] = dict(retry_args) if isinstance(retry_args, dict) else {}
            return out
        except Exception:
            return fallback

    def _reflect_on_required_failure(
        self,
        *,
        node: PlanNode,
        rec: Dict[str, Any],
        binder: _ScopedBinder,
        scope_domain: str,
        case_id: str,
        run_id: str,
    ) -> Dict[str, Any]:
        exec_tool_name = self._effective_tool_name(node)
        last_arguments = (
            dict(rec.get("arguments") or {})
            if isinstance(rec.get("arguments"), dict)
            else dict(node.arguments or {})
        )
        err = dict(rec.get("error") or {}) if isinstance(rec.get("error"), dict) else {}
        binding_context = self._collect_reflection_binding_context(binder=binder)
        deterministic_retry = self._deterministic_retry_suggestion(
            node=node,
            last_args=last_arguments,
            error=err,
            binding_context=binding_context,
        )
        payload = {
            "node_id": str(node.node_id),
            "tool_name": str(exec_tool_name),
            "stage": str(node.stage or ""),
            "scope_domain": str(scope_domain),
            "case_id": str(case_id),
            "run_id": str(run_id),
            "error": err,
            "last_arguments": last_arguments,
            "status": str(rec.get("status") or "FAIL"),
            "binding_context": binding_context,
            "deterministic_retry_suggestion": deterministic_retry,
        }
        verdict = self._failure_reflect_fn(payload)
        if not isinstance(verdict, dict):
            verdict = {}
        action = str(verdict.get("action") or "halt").strip().lower()
        if action not in {"halt", "skip", "retry"}:
            action = "halt"
        reason = str(verdict.get("reason") or "").strip()
        nl = str(verdict.get("natural_language_response") or "").strip()
        retry_args = verdict.get("retry_arguments")
        if not isinstance(retry_args, dict):
            retry_args = {}
        retry_args, _ = self._apply_ref_token_aliases(dict(retry_args))
        if action == "retry" and not retry_args and deterministic_retry:
            retry_args = dict(deterministic_retry)
        if action in {"halt", "skip"} and deterministic_retry:
            action = "retry"
            retry_args = dict(deterministic_retry)
            if not reason:
                reason = "Auto-corrected unresolved path/token binding for retry."
            if not nl:
                nl = "I corrected a path/token binding issue and retried the failed required step."
        return {
            "action": action,
            "reason": reason,
            "natural_language_response": nl,
            "retry_arguments": dict(retry_args),
        }

    def execute_plan(
        self,
        plan: Dict[str, Any],
        *,
        emit: Callable[[str], None] = print,
        start_from_node_id: Optional[str] = None,
        invalidate_downstream: bool = True,
        node_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        plan_obj = dict(plan or {})
        domain = str(plan_obj.get("domain") or "prostate").strip().lower() or "prostate"
        self._sync_case_from_plan(plan_obj)

        case_id = str(
            plan_obj.get("case_id")
            or self.session.case_id
            or f"case_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        ).strip()
        self.session.case_id = case_id

        case_ref = str(plan_obj.get("case_ref") or _guess_case_ref_from_session(self.session) or "").strip()
        if case_ref in {"", "case.input", "@case.input", "case_input", "-"}:
            case_ref = _guess_case_ref_from_session(self.session)
        if not case_ref:
            raise ValueError("Cannot execute plan without case_ref bound to a real directory")
        p = Path(case_ref).expanduser()
        if not p.is_absolute():
            p = Path(self.session.workspace_path) / p
        case_ref = str(_safe_abspath(str(p)))

        dag = legacy_plan_to_dag(
            legacy_plan=plan_obj,
            domain=domain,
            case_id=case_id,
            case_ref=case_ref,
            workspace_root=str(self.session.workspace_path),
            runs_root=str(self.session.runs_root),
            plan_id=str(plan_obj.get("plan_id") or plan_obj.get("run_id") or ""),
        )
        return self.execute_dag(
            dag,
            emit=emit,
            start_from_node_id=start_from_node_id,
            invalidate_downstream=invalidate_downstream,
            node_overrides=node_overrides,
            event_sink=event_sink,
        )

    def execute_dag(
        self,
        dag: AgentPlanDAG | Dict[str, Any],
        *,
        emit: Callable[[str], None] = print,
        start_from_node_id: Optional[str] = None,
        invalidate_downstream: bool = True,
        node_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        dag_obj = AgentPlanDAG.model_validate(dag)
        if node_overrides:
            dag_obj = self._apply_node_overrides(dag_obj, node_overrides=node_overrides)
        active_nodes = self._active_nodes_for_rerun(
            dag_obj,
            start_from_node_id=start_from_node_id,
            invalidate_downstream=bool(invalidate_downstream),
        )
        scope = dag_obj.case_scope

        case_root = _safe_abspath(scope.case_ref)
        if not case_root.exists() or not case_root.is_dir():
            raise ValueError(f"case_ref is not a directory: {case_root}")

        self._reset_session_scope(scope_key=f"{scope.domain}|{case_root}", case_root=case_root, case_id=scope.case_id)

        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        if event_sink is not None:
            self.dispatcher.set_event_sink(event_sink)
        state, ctx = self.dispatcher.create_run(case_id=scope.case_id, run_id=run_id)

        try:
            state.metadata = {**(state.metadata or {}), "domain": str(scope.domain)}
            state.write_json(ctx.case_state_path)
        except Exception:
            pass

        trace_path = Path(self.session.workspace_path) / "logs" / scope.case_id / "tool_trace.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        binder = _ScopedBinder(case_root=case_root)
        binder.seed_runtime(ctx)

        guard = _CaseScopeGuard(
            case_root=case_root,
            run_root=Path(ctx.run_dir).resolve(),
            external_roots=[
                *[
                    _safe_resolve(x)
                    for x in (scope.allow_external_model_roots or [])
                    if str(x).strip()
                ],
                *_default_external_model_roots(),
            ],
        )

        step_results: List[Dict[str, Any]] = []
        reflection_decisions: List[Dict[str, Any]] = []
        node_status: Dict[str, str] = {}
        node_by_id = {n.node_id: n for n in dag_obj.nodes}
        overall_ok = True
        halt = False
        agent_response = ""

        while len(node_status) < len(dag_obj.nodes):
            progressed = False

            for node in dag_obj.nodes:
                node_id = str(node.node_id)
                if node_id in node_status:
                    continue

                if active_nodes is not None and node_id not in active_nodes:
                    rec = self._skipped_step_record(
                        node=node,
                        reason=f"outside rerun scope (start_from={start_from_node_id})",
                        status="SKIPPED_RERUN_SCOPE",
                        case_id=scope.case_id,
                        run_id=run_id,
                        trace_path=trace_path,
                        emit=emit,
                    )
                    step_results.append(rec)
                    node_status[node_id] = "SKIPPED_RERUN_SCOPE"
                    progressed = True
                    continue

                if bool(getattr(node, "skip", False)):
                    rec = self._skipped_step_record(
                        node=node,
                        reason="skipped by node override",
                        status="SKIPPED_PATCH",
                        case_id=scope.case_id,
                        run_id=run_id,
                        trace_path=trace_path,
                        emit=emit,
                    )
                    step_results.append(rec)
                    node_status[node_id] = "SKIPPED_PATCH"
                    progressed = True
                    continue

                if (not bool(node.required)) and bool(dag_obj.policy.skip_optional_by_default):
                    rec = self._skipped_step_record(
                        node=node,
                        reason="optional node skipped by policy (required=False)",
                        status="SKIPPED_OPTIONAL",
                        case_id=scope.case_id,
                        run_id=run_id,
                        trace_path=trace_path,
                        emit=emit,
                    )
                    step_results.append(rec)
                    node_status[node_id] = "SKIPPED_OPTIONAL"
                    progressed = True
                    continue

                deps = [d for d in node.depends_on if str(d).strip()]
                unresolved = [d for d in deps if d not in node_status]
                if unresolved:
                    continue

                bad_deps = [d for d in deps if node_status.get(d) != "DONE"]
                if bad_deps:
                    bad_desc = ", ".join([f"{d}:{node_status.get(d)}" for d in bad_deps])
                    status = "BLOCKED" if node.required else "SKIPPED_OPTIONAL"
                    rec = self._skipped_step_record(
                        node=node,
                        reason=f"blocked by dependencies: {bad_desc}",
                        status=status,
                        case_id=scope.case_id,
                        run_id=run_id,
                        trace_path=trace_path,
                        emit=emit,
                    )
                    step_results.append(rec)
                    node_status[node_id] = status
                    progressed = True
                    if node.required:
                        overall_ok = False
                        if bool(dag_obj.policy.stop_on_required_failure):
                            halt = True
                            break
                    continue

                ok, rec = self._execute_node(
                    node=node,
                    state=state,
                    ctx=ctx,
                    scope_domain=str(scope.domain),
                    guard=guard,
                    binder=binder,
                    case_id=scope.case_id,
                    run_id=run_id,
                    trace_path=trace_path,
                    max_attempts=(node.max_attempts or dag_obj.policy.max_attempts_default or self.max_attempts),
                    emit=emit,
                )
                step_results.append(rec)
                node_status[node_id] = str(rec.get("status") or ("DONE" if ok else "FAIL"))
                if ok:
                    step_answer = self._extract_natural_response_from_step(rec)
                    if step_answer:
                        agent_response = step_answer
                progressed = True
                if not ok and node.required:
                    reflection = self._reflect_on_required_failure(
                        node=node,
                        rec=rec,
                        binder=binder,
                        scope_domain=str(scope.domain),
                        case_id=scope.case_id,
                        run_id=run_id,
                    )
                    action = str(reflection.get("action") or "halt").strip().lower()
                    reason = str(reflection.get("reason") or "").strip()
                    nl = str(reflection.get("natural_language_response") or "").strip()
                    reflection_rec = {
                        "node_id": node_id,
                        "tool_name": self._effective_tool_name(node),
                        "action": action,
                        "reason": reason,
                        "natural_language_response": nl,
                    }
                    reflection_decisions.append(reflection_rec)
                    append_event(
                        trace_path,
                        event_record(
                            event_type="reflection",
                            tool_name=self._effective_tool_name(node),
                            status=action.upper(),
                            case_id=scope.case_id,
                            run_id=run_id,
                            error=({"type": "ReflectionDecision", "message": reason} if reason else None),
                            outputs={"natural_language_response": nl} if nl else None,
                        ),
                    )
                    if nl:
                        agent_response = nl
                        emit(f"REFLECT {node.node_id} {self._effective_tool_name(node)}: {nl}")

                    if action == "retry":
                        retry_args = reflection.get("retry_arguments")
                        if not isinstance(retry_args, dict):
                            retry_args = {}
                        retry_node = node.model_copy(
                            update={
                                "required": True,
                                "arguments": (retry_args if retry_args else dict(node.arguments or {})),
                            }
                        )
                        ok_retry, rec_retry = self._execute_node(
                            node=retry_node,
                            state=state,
                            ctx=ctx,
                            scope_domain=str(scope.domain),
                            guard=guard,
                            binder=binder,
                            case_id=scope.case_id,
                            run_id=run_id,
                            trace_path=trace_path,
                            max_attempts=self._failure_retry_limit,
                            emit=emit,
                        )
                        rec_retry["reflection_action"] = "retry"
                        step_results.append(rec_retry)
                        node_status[node_id] = str(rec_retry.get("status") or ("DONE" if ok_retry else "FAIL"))
                        if ok_retry:
                            retry_answer = self._extract_natural_response_from_step(rec_retry)
                            if retry_answer:
                                agent_response = retry_answer
                        if ok_retry:
                            continue
                        overall_ok = False
                        if bool(dag_obj.policy.stop_on_required_failure):
                            halt = True
                            break
                        continue

                    if action == "skip":
                        node_status[node_id] = "SKIPPED_BY_REFLECTION"
                        if isinstance(step_results[-1], dict):
                            step_results[-1]["status"] = "SKIPPED_BY_REFLECTION"
                            step_results[-1]["reflection_action"] = "skip"
                        overall_ok = False
                        continue

                    overall_ok = False
                    if bool(dag_obj.policy.stop_on_required_failure):
                        halt = True
                        break

            if halt:
                break

            if not progressed:
                # Cycle/invalid dependency references: mark remaining nodes blocked.
                for node in dag_obj.nodes:
                    node_id = str(node.node_id)
                    if node_id in node_status:
                        continue
                    missing = [d for d in node.depends_on if d not in node_by_id and d not in node_status]
                    waiting = [d for d in node.depends_on if d in node_by_id and d not in node_status]
                    reason_bits = []
                    if missing:
                        reason_bits.append(f"unknown deps={missing}")
                    if waiting:
                        reason_bits.append(f"cyclic/unreachable deps={waiting}")
                    reason = "; ".join(reason_bits) or "unreachable dependency state"
                    status = "BLOCKED" if node.required else "SKIPPED_OPTIONAL"
                    rec = self._skipped_step_record(
                        node=node,
                        reason=reason,
                        status=status,
                        case_id=scope.case_id,
                        run_id=run_id,
                        trace_path=trace_path,
                        emit=emit,
                    )
                    step_results.append(rec)
                    node_status[node_id] = status
                    if node.required:
                        overall_ok = False
                break

        report_path = ensure_report(
            workspace_path=Path(self.session.workspace_path),
            case_id=scope.case_id,
            goal=dag_obj.goal,
            run_dir=ctx.run_dir,
            step_results=step_results,
        )

        final_event = event_record(
            event_type="session_end",
            tool_name="<session>",
            status=("DONE" if overall_ok else "FAIL"),
            case_id=scope.case_id,
            run_id=run_id,
            outputs={
                "report_path": str(report_path),
                "run_dir": str(ctx.run_dir),
                "scope_id": str(scope.scope_id),
            },
        )
        append_event(trace_path, final_event)

        self.session.last_run = {
            "ok": overall_ok,
            "case_id": scope.case_id,
            "run_id": run_id,
            "run_dir": str(ctx.run_dir),
            "trace_path": str(trace_path),
            "report_path": str(report_path),
            "domain": str(scope.domain),
            "scope_id": str(scope.scope_id),
            "n_steps": len(step_results),
            "start_from_node_id": (str(start_from_node_id) if start_from_node_id else None),
            "active_node_count": (len(active_nodes) if active_nodes is not None else len(dag_obj.nodes)),
            "reflection_decisions": reflection_decisions,
            "natural_language_response": agent_response,
            "final_answer": agent_response,
        }
        self.session.set_path_key("report.md", str(report_path))
        self.session.set_path_key("runtime.last_run_dir", str(ctx.run_dir))
        self.session.set_path_key("runtime.last_trace", str(trace_path))

        return dict(self.session.last_run)

    def _effective_tool_name(self, node: PlanNode) -> str:
        locked = str(getattr(node, "tool_locked", "") or "").strip()
        if locked:
            return locked
        selected = str(getattr(node, "tool_selected", "") or "").strip()
        if selected:
            return selected
        return str(node.tool_name or "")

    def _apply_node_overrides(
        self,
        dag_obj: AgentPlanDAG,
        *,
        node_overrides: Dict[str, Dict[str, Any]],
    ) -> AgentPlanDAG:
        if not node_overrides:
            return dag_obj
        override_map = {str(k): dict(v or {}) for k, v in (node_overrides or {}).items()}
        new_nodes: List[PlanNode] = []
        for node in dag_obj.nodes:
            ov = override_map.get(str(node.node_id))
            if not ov:
                new_nodes.append(node)
                continue
            update: Dict[str, Any] = {}
            if "tool_locked" in ov:
                v = str(ov.get("tool_locked") or "").strip()
                update["tool_locked"] = (v or None)
            if isinstance(ov.get("config_values"), dict):
                cv = dict(node.config_values or {})
                cv.update(dict(ov.get("config_values") or {}))
                update["config_values"] = cv
            if isinstance(ov.get("config_locked_fields"), list):
                update["config_locked_fields"] = [str(x) for x in ov.get("config_locked_fields") if str(x).strip()]
            if "skip" in ov and ov.get("skip") is not None:
                update["skip"] = bool(ov.get("skip"))
            new_nodes.append(node.model_copy(update=update))
        return dag_obj.model_copy(update={"nodes": new_nodes})

    def _active_nodes_for_rerun(
        self,
        dag_obj: AgentPlanDAG,
        *,
        start_from_node_id: Optional[str],
        invalidate_downstream: bool,
    ) -> Optional[set[str]]:
        start = str(start_from_node_id or "").strip()
        if not start:
            return None
        node_ids = {str(n.node_id) for n in dag_obj.nodes}
        if start not in node_ids:
            raise ValueError(f"start_from_node_id not found in DAG: {start}")

        rev: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        fwd: Dict[str, List[str]] = {nid: [] for nid in node_ids}
        for node in dag_obj.nodes:
            nid = str(node.node_id)
            for dep in [str(d) for d in (node.depends_on or []) if str(d).strip()]:
                if dep in node_ids:
                    rev[nid].append(dep)
                    fwd[dep].append(nid)

        ancestors: set[str] = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            for p in rev.get(cur, []):
                if p in ancestors:
                    continue
                ancestors.add(p)
                stack.append(p)

        descendants: set[str] = set()
        if invalidate_downstream:
            queue = [start]
            while queue:
                cur = queue.pop(0)
                for nxt in fwd.get(cur, []):
                    if nxt in descendants:
                        continue
                    descendants.add(nxt)
                    queue.append(nxt)

        return ancestors | {start} | descendants

    def _reset_session_scope(self, *, scope_key: str, case_root: Path, case_id: str) -> None:
        if self._last_scope_key != scope_key:
            self.session.path_keys = {}
            self._last_scope_key = scope_key
        self.session.case_inputs["case_input_path"] = str(case_root)
        self.session.path_keys["case.input"] = str(case_root)
        self.session.case_id = str(case_id)

    def _skipped_step_record(
        self,
        *,
        node: PlanNode,
        reason: str,
        status: str,
        case_id: str,
        run_id: str,
        trace_path: Path,
        emit: Callable[[str], None],
    ) -> Dict[str, Any]:
        args = dict(node.arguments or {})
        tool_name = self._effective_tool_name(node)
        args_summary = summarize_args(args)
        err = {"type": "Skipped", "message": reason}
        rec = event_record(
            event_type="tool_call",
            tool_name=str(tool_name),
            status=status,
            case_id=case_id,
            run_id=run_id,
            args_summary=args_summary,
            error=err,
        )
        rec["node_id"] = str(node.node_id)
        rec["label"] = str(node.label or "")
        append_event(trace_path, rec)
        emit(f"SKIP {node.node_id} {tool_name}: {reason}")
        return {
            "node_id": str(node.node_id),
            "label": str(node.label or ""),
            "tool_name": str(tool_name),
            "tool_selected": str(getattr(node, "tool_selected", "") or ""),
            "tool_locked": (str(getattr(node, "tool_locked", "") or "").strip() or None),
            "tool_executed": str(tool_name),
            "status": status,
            "required": bool(node.required),
            "arguments": dict(args),
            "config_values": dict(getattr(node, "config_values", {}) or {}),
            "duration_s": None,
            "error": err,
            "outputs": {},
            "data": {},
        }

    def _execute_node(
        self,
        *,
        node: PlanNode,
        state: CaseState,
        ctx: ToolContext,
        scope_domain: str,
        guard: _CaseScopeGuard,
        binder: _ScopedBinder,
        case_id: str,
        run_id: str,
        trace_path: Path,
        max_attempts: int,
        emit: Callable[[str], None],
    ) -> Tuple[bool, Dict[str, Any]]:
        node_type = str(getattr(node, "node_type", "tool") or "tool").strip().lower() or "tool"
        if node_type in {"reflect", "gate"}:
            return self._execute_control_node(
                node=node,
                node_type=node_type,
                state=state,
                ctx=ctx,
                scope_domain=scope_domain,
                case_id=case_id,
                run_id=run_id,
                trace_path=trace_path,
                emit=emit,
            )

        tool_name = self._effective_tool_name(node)
        stage = str(node.stage or "misc")
        required = bool(node.required)
        current_args = dict(node.arguments or {})
        if isinstance(getattr(node, "config_values", None), dict):
            current_args.update(dict(node.config_values or {}))
        attempts = max(1, int(max_attempts or 1))
        last_error: Optional[Dict[str, Any]] = None

        for attempt in range(1, attempts + 1):
            resolved_args = binder.resolve_refs(current_args)
            resolved_args = self._normalize_step_args(tool_name=tool_name, args=resolved_args, binder=binder)
            repaired_args = repair_tool_args(
                tool_name,
                dict(resolved_args),
                state_path=ctx.case_state_path,
                ctx_case_state_path=ctx.case_state_path,
                dicom_case_dir=self.session.case_inputs.get("case_input_path"),
                domain=get_domain_config(scope_domain),
            )
            repaired_args = self._normalize_step_args(tool_name=tool_name, args=repaired_args, binder=binder)

            try:
                guard.validate_args(tool_name=tool_name, args=repaired_args)
            except ScopeViolation as e:
                err = {"type": "ScopeViolation", "message": str(e)}
                args_summary = summarize_args(repaired_args)
                fail = event_record(
                    event_type="tool_call",
                    tool_name=tool_name,
                    status="FAIL",
                    case_id=case_id,
                    run_id=run_id,
                    args_summary=args_summary,
                    error=err,
                    attempt=attempt,
                )
                fail["node_id"] = str(node.node_id)
                fail["label"] = str(node.label or "")
                append_event(trace_path, fail)
                emit(f"FAIL {node.node_id} {tool_name}: {err['message']}")
                return False, {
                    "node_id": str(node.node_id),
                    "label": str(node.label or ""),
                    "tool_name": tool_name,
                    "tool_selected": str(getattr(node, "tool_selected", "") or ""),
                    "tool_locked": (str(getattr(node, "tool_locked", "") or "").strip() or None),
                    "tool_executed": str(tool_name),
                    "status": "FAIL",
                    "required": required,
                    "arguments": dict(repaired_args),
                    "config_values": dict(getattr(node, "config_values", {}) or {}),
                    "duration_s": None,
                    "error": err,
                    "outputs": {},
                    "data": {},
                }

            args_summary = summarize_args(repaired_args)
            start = event_record(
                event_type="tool_call",
                tool_name=tool_name,
                status="RUNNING",
                case_id=case_id,
                run_id=run_id,
                args_summary=args_summary,
                attempt=attempt,
            )
            start["node_id"] = str(node.node_id)
            start["label"] = str(node.label or "")
            append_event(trace_path, start)
            emit(f"RUN {node.node_id} {tool_name}({args_summary})")

            t0 = time.time()
            try:
                self._validate_tool_args(tool_name=tool_name, args=repaired_args)
                call = ToolCall(
                    tool_name=tool_name,
                    arguments=repaired_args,
                    call_id=f"dag_{node.node_id}_try{attempt}",
                    case_id=case_id,
                    stage=stage,
                    requested_by="mri_agent_shell:cerebellum",
                )
                result = self.dispatcher.dispatch(call, state, ctx)
                dur = time.time() - t0

                if result.ok:
                    outputs = summarize_outputs(result.data)
                    self._learn_from_result(tool_name=tool_name, data=result.data, binder=binder)
                    binder.learn_from_node(node_id=str(node.node_id), data=result.data)
                    end = event_record(
                        event_type="tool_call",
                        tool_name=tool_name,
                        status="DONE",
                        case_id=case_id,
                        run_id=run_id,
                        args_summary=args_summary,
                        duration_s=dur,
                        outputs=outputs,
                        attempt=attempt,
                    )
                    end["node_id"] = str(node.node_id)
                    end["label"] = str(node.label or "")
                    append_event(trace_path, end)
                    emit(f"DONE {node.node_id} {tool_name}: {dur:.2f}s")
                    return True, {
                        "node_id": str(node.node_id),
                        "label": str(node.label or ""),
                        "tool_name": tool_name,
                        "tool_selected": str(getattr(node, "tool_selected", "") or ""),
                        "tool_locked": (str(getattr(node, "tool_locked", "") or "").strip() or None),
                        "tool_executed": str(tool_name),
                        "status": "DONE",
                        "required": required,
                        "arguments": dict(repaired_args),
                        "config_values": dict(getattr(node, "config_values", {}) or {}),
                        "duration_s": round(dur, 4),
                        "outputs": outputs,
                        "data": result.data,
                    }

                err = result.error.to_dict() if result.error else {"type": "RuntimeError", "message": "unknown error"}
                last_error = err
                fail = event_record(
                    event_type="tool_call",
                    tool_name=tool_name,
                    status="FAIL",
                    case_id=case_id,
                    run_id=run_id,
                    args_summary=args_summary,
                    duration_s=dur,
                    error=err,
                    attempt=attempt,
                )
                fail["node_id"] = str(node.node_id)
                fail["label"] = str(node.label or "")
                append_event(trace_path, fail)
                emit(f"FAIL {node.node_id} {tool_name}: {err.get('type')}: {err.get('message')}")

                if attempt < attempts and self._is_retryable(err):
                    current_args = self._repair_args_after_error(tool_name=tool_name, args=current_args, err=err, binder=binder)
                    retry = event_record(
                        event_type="retry",
                        tool_name=tool_name,
                        status="RETRYING",
                        case_id=case_id,
                        run_id=run_id,
                        args_summary=args_summary,
                        error=err,
                        attempt=attempt,
                    )
                    retry["node_id"] = str(node.node_id)
                    retry["label"] = str(node.label or "")
                    append_event(trace_path, retry)
                    continue

                return False, {
                    "node_id": str(node.node_id),
                    "label": str(node.label or ""),
                    "tool_name": tool_name,
                    "tool_selected": str(getattr(node, "tool_selected", "") or ""),
                    "tool_locked": (str(getattr(node, "tool_locked", "") or "").strip() or None),
                    "tool_executed": str(tool_name),
                    "status": "FAIL",
                    "required": required,
                    "arguments": dict(repaired_args),
                    "config_values": dict(getattr(node, "config_values", {}) or {}),
                    "duration_s": round(dur, 4),
                    "error": err,
                    "outputs": {},
                    "data": {},
                }

            except Exception as e:
                dur = time.time() - t0
                err = {"type": type(e).__name__, "message": str(e)}
                last_error = err
                fail = event_record(
                    event_type="tool_call",
                    tool_name=tool_name,
                    status="FAIL",
                    case_id=case_id,
                    run_id=run_id,
                    args_summary=args_summary,
                    duration_s=dur,
                    error=err,
                    attempt=attempt,
                )
                fail["node_id"] = str(node.node_id)
                fail["label"] = str(node.label or "")
                append_event(trace_path, fail)
                emit(f"FAIL {node.node_id} {tool_name}: {err['type']}: {err['message']}")

                if attempt < attempts and self._is_retryable(err):
                    current_args = self._repair_args_after_error(tool_name=tool_name, args=current_args, err=err, binder=binder)
                    retry = event_record(
                        event_type="retry",
                        tool_name=tool_name,
                        status="RETRYING",
                        case_id=case_id,
                        run_id=run_id,
                        args_summary=args_summary,
                        error=err,
                        attempt=attempt,
                    )
                    retry["node_id"] = str(node.node_id)
                    retry["label"] = str(node.label or "")
                    append_event(trace_path, retry)
                    continue

                return False, {
                    "node_id": str(node.node_id),
                    "label": str(node.label or ""),
                    "tool_name": tool_name,
                    "tool_selected": str(getattr(node, "tool_selected", "") or ""),
                    "tool_locked": (str(getattr(node, "tool_locked", "") or "").strip() or None),
                    "tool_executed": str(tool_name),
                    "status": "FAIL",
                    "required": required,
                    "arguments": dict(repaired_args),
                    "config_values": dict(getattr(node, "config_values", {}) or {}),
                    "duration_s": round(dur, 4),
                    "error": err,
                    "outputs": {},
                    "data": {},
                }

        return False, {
            "node_id": str(node.node_id),
            "label": str(node.label or ""),
            "tool_name": tool_name,
            "tool_selected": str(getattr(node, "tool_selected", "") or ""),
            "tool_locked": (str(getattr(node, "tool_locked", "") or "").strip() or None),
            "tool_executed": str(tool_name),
            "status": "FAIL",
            "required": required,
            "arguments": dict(current_args),
            "config_values": dict(getattr(node, "config_values", {}) or {}),
            "duration_s": None,
            "error": last_error or {"type": "RuntimeError", "message": "exhausted retries"},
            "outputs": {},
            "data": {},
        }

    def _execute_control_node(
        self,
        *,
        node: PlanNode,
        node_type: str,
        state: CaseState,
        ctx: ToolContext,
        scope_domain: str,
        case_id: str,
        run_id: str,
        trace_path: Path,
        emit: Callable[[str], None],
    ) -> Tuple[bool, Dict[str, Any]]:
        del state  # reserved for future control-node policies
        tool_name = self._effective_tool_name(node) or str(node.tool_name or f"__{node_type}__")
        args = dict(node.arguments or {})
        if isinstance(getattr(node, "config_values", None), dict):
            args.update(dict(node.config_values or {}))
        args_summary = summarize_args(args)
        start = event_record(
            event_type="control_node",
            tool_name=tool_name,
            status="RUNNING",
            case_id=case_id,
            run_id=run_id,
            args_summary=args_summary,
        )
        start["node_id"] = str(node.node_id)
        start["label"] = str(node.label or "")
        append_event(trace_path, start)
        emit(f"RUN {node.node_id} {tool_name}({args_summary}) [node_type={node_type}]")

        t0 = time.time()
        payload = {
            "node_id": str(node.node_id),
            "node_type": node_type,
            "tool_name": tool_name,
            "label": str(node.label or ""),
            "arguments": args,
            "scope_domain": str(scope_domain),
            "case_id": str(case_id),
            "run_id": str(run_id),
            "state_path": str(ctx.case_state_path),
            "run_dir": str(ctx.run_dir),
        }
        try:
            verdict = self._reflect_fn(payload)
            decision = str((verdict or {}).get("decision") or "continue").strip().lower()
            if decision not in {"continue", "halt"}:
                raise ValueError(f"unsupported control decision: {decision}")
            reason = str((verdict or {}).get("reason") or "").strip()
            dur = time.time() - t0

            if decision == "halt":
                err = {"type": "ReflectHalt", "message": reason or "control node requested halt"}
                fail = event_record(
                    event_type="control_node",
                    tool_name=tool_name,
                    status="FAIL",
                    case_id=case_id,
                    run_id=run_id,
                    args_summary=args_summary,
                    duration_s=dur,
                    error=err,
                )
                fail["node_id"] = str(node.node_id)
                fail["label"] = str(node.label or "")
                append_event(trace_path, fail)
                emit(f"FAIL {node.node_id} {tool_name}: {err['message']}")
                return False, {
                    "node_id": str(node.node_id),
                    "label": str(node.label or ""),
                    "tool_name": tool_name,
                    "tool_selected": str(getattr(node, "tool_selected", "") or ""),
                    "tool_locked": (str(getattr(node, "tool_locked", "") or "").strip() or None),
                    "tool_executed": str(tool_name),
                    "status": "FAIL",
                    "required": bool(node.required),
                    "arguments": dict(args),
                    "config_values": dict(getattr(node, "config_values", {}) or {}),
                    "duration_s": round(dur, 4),
                    "error": err,
                    "outputs": {},
                    "data": {"decision": decision, **({"reason": reason} if reason else {})},
                }

            outputs = {"decision": decision, **({"reason": reason} if reason else {})}
            done = event_record(
                event_type="control_node",
                tool_name=tool_name,
                status="DONE",
                case_id=case_id,
                run_id=run_id,
                args_summary=args_summary,
                duration_s=dur,
                outputs=outputs,
            )
            done["node_id"] = str(node.node_id)
            done["label"] = str(node.label or "")
            append_event(trace_path, done)
            emit(f"DONE {node.node_id} {tool_name}: {dur:.2f}s")
            return True, {
                "node_id": str(node.node_id),
                "label": str(node.label or ""),
                "tool_name": tool_name,
                "tool_selected": str(getattr(node, "tool_selected", "") or ""),
                "tool_locked": (str(getattr(node, "tool_locked", "") or "").strip() or None),
                "tool_executed": str(tool_name),
                "status": "DONE",
                "required": bool(node.required),
                "arguments": dict(args),
                "config_values": dict(getattr(node, "config_values", {}) or {}),
                "duration_s": round(dur, 4),
                "outputs": outputs,
                "data": outputs,
            }
        except Exception as e:
            dur = time.time() - t0
            err = {"type": type(e).__name__, "message": str(e)}
            fail = event_record(
                event_type="control_node",
                tool_name=tool_name,
                status="FAIL",
                case_id=case_id,
                run_id=run_id,
                args_summary=args_summary,
                duration_s=dur,
                error=err,
            )
            fail["node_id"] = str(node.node_id)
            fail["label"] = str(node.label or "")
            append_event(trace_path, fail)
            emit(f"FAIL {node.node_id} {tool_name}: {err['type']}: {err['message']}")
            return False, {
                "node_id": str(node.node_id),
                "label": str(node.label or ""),
                "tool_name": tool_name,
                "tool_selected": str(getattr(node, "tool_selected", "") or ""),
                "tool_locked": (str(getattr(node, "tool_locked", "") or "").strip() or None),
                "tool_executed": str(tool_name),
                "status": "FAIL",
                "required": bool(node.required),
                "arguments": dict(args),
                "config_values": dict(getattr(node, "config_values", {}) or {}),
                "duration_s": round(dur, 4),
                "error": err,
                "outputs": {},
                "data": {},
            }

    def _sync_case_from_plan(self, plan: Dict[str, Any]) -> None:
        if not isinstance(plan, dict):
            return
        raw_ref = str(plan.get("case_ref") or "").strip()
        if not raw_ref:
            return

        if raw_ref in {"case.input", "@case.input"}:
            bound = str(self.session.case_inputs.get("case_input_path") or "").strip()
            if bound:
                self.session.path_keys["case.input"] = bound
            return

        p = Path(raw_ref).expanduser()
        if not p.is_absolute():
            p = Path(self.session.workspace_path) / p
        try:
            case_ref = str(_safe_abspath(str(p)))
        except Exception:
            case_ref = str(p)

        self.session.case_inputs["case_input_path"] = case_ref
        self.session.path_keys["case.input"] = case_ref
        if not str(self.session.case_id or "").strip():
            self.session.case_id = Path(case_ref).name

    def _extract_natural_response_from_step(self, rec: Dict[str, Any]) -> str:
        if not isinstance(rec, dict):
            return ""
        data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
        for key in ("final_answer", "natural_language_response", "answer"):
            v = str(data.get(key) or "").strip()
            if v:
                return v
        return ""

    def _inject_runtime_llm_args(self, *, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(args or {})
        if tool_name not in {"generate_report", "rag_search"}:
            return out

        cfg = self.session.model_config
        provider = str(getattr(cfg, "provider", "") or "").strip().lower()
        model_name = str(getattr(cfg, "llm", "") or "").strip()
        server_base_url = str(getattr(cfg, "server_base_url", "") or "").strip()
        api_base_url = str(getattr(cfg, "api_base_url", "") or "").strip()
        max_tokens = int(getattr(cfg, "max_tokens", 0) or 0)
        temperature = float(getattr(cfg, "temperature", 0.0) or 0.0)

        llm_mode = str(out.get("llm_mode") or "").strip().lower()
        if not llm_mode:
            provider_to_mode = {
                "stub": "stub",
                "openai_compatible_server": "server",
                "openai_official": "openai",
                "anthropic": "anthropic",
                "gemini": "gemini",
            }
            llm_mode = provider_to_mode.get(provider, "")
            if llm_mode:
                out["llm_mode"] = llm_mode

        if llm_mode == "server":
            if not str(out.get("server_base_url") or "").strip() and server_base_url:
                out["server_base_url"] = server_base_url
            if not str(out.get("server_model") or "").strip() and model_name:
                out["server_model"] = model_name
        elif llm_mode in {"openai", "anthropic", "gemini"}:
            if not str(out.get("api_model") or "").strip() and model_name:
                out["api_model"] = model_name
            if llm_mode == "openai" and not str(out.get("api_base_url") or "").strip() and api_base_url:
                out["api_base_url"] = api_base_url

        try:
            cur_max_tokens = int(out.get("max_tokens") or 0)
        except Exception:
            cur_max_tokens = 0
        if cur_max_tokens <= 0 and max_tokens > 0:
            out["max_tokens"] = max_tokens

        temp_val = out.get("temperature")
        if temp_val is None or (isinstance(temp_val, str) and not temp_val.strip()):
            out["temperature"] = temperature
        return out

    def _normalize_step_args(
        self,
        *,
        tool_name: str,
        args: Dict[str, Any],
        binder: Optional[_ScopedBinder] = None,
    ) -> Dict[str, Any]:
        out = dict(args or {})
        bind = binder
        if bind is None:
            case_ref = _guess_case_ref_from_session(self.session)
            if not case_ref:
                return out
            try:
                bind = _ScopedBinder(case_root=_safe_abspath(case_ref))
            except Exception:
                bind = _ScopedBinder(case_root=Path(case_ref).expanduser())

        out = self._inject_runtime_llm_args(tool_name=tool_name, args=out)

        if tool_name == "identify_sequences":
            raw = out.get("dicom_case_dir")
            if isinstance(raw, str):
                case_input = str(bind.case_root)
                if raw.startswith("@") or raw in {"case.input", "@case.input"}:
                    out["dicom_case_dir"] = case_input
                else:
                    try:
                        p = Path(raw).expanduser()
                        if not p.is_absolute():
                            out["dicom_case_dir"] = case_input
                    except Exception:
                        out["dicom_case_dir"] = case_input
            return out

        if tool_name == "segment_prostate":
            tok = str(out.get("t2w_ref") or "").strip()
            if tok:
                resolved = bind.resolve_sequence_or_case_path(tok, preferred_tokens=["T2w", "T2"], prefer_file=True)
                if resolved:
                    out["t2w_ref"] = resolved
            return out

        if tool_name == "register_to_reference":
            fixed = str(out.get("fixed") or "").strip()
            moving = str(out.get("moving") or "").strip()
            if fixed:
                fixed_res = bind.resolve_sequence_or_case_path(
                    fixed,
                    preferred_tokens=["T2w", "T1c", "T1", "CINE"],
                    prefer_file=True,
                )
                if fixed_res:
                    out["fixed"] = fixed_res
            if moving:
                move_res = bind.resolve_sequence_or_case_path(
                    moving,
                    preferred_tokens=["ADC", "DWI", "FLAIR", "T2", "CINE"],
                    prefer_file=True,
                )
                if move_res:
                    out["moving"] = move_res
            return out

        if tool_name == "brats_mri_segmentation":
            for key, prefs in (
                ("t1c_path", ["T1c", "T1"]),
                ("t1_path", ["T1"]),
                ("t2_path", ["T2"]),
                ("flair_path", ["FLAIR"]),
            ):
                raw = str(out.get(key) or "").strip()
                if not raw:
                    continue
                resolved = bind.resolve_sequence_or_case_path(raw, preferred_tokens=prefs, prefer_file=True)
                if resolved:
                    out[key] = resolved
            return out

        if tool_name == "segment_cardiac_cine":
            raw = str(out.get("cine_path") or "").strip()
            if raw:
                resolved = bind.resolve_sequence_or_case_path(raw, preferred_tokens=["CINE"], prefer_file=True)
                if resolved:
                    out["cine_path"] = resolved
            return out

        if tool_name == "dummy_load_case":
            raw = str(out.get("case_path") or "").strip()
            if raw in {"case.input", "@case.input"}:
                out["case_path"] = str(bind.case_root)

        return out

    def _repair_args_after_error(
        self,
        *,
        tool_name: str,
        args: Dict[str, Any],
        err: Dict[str, Any],
        binder: Optional[_ScopedBinder] = None,
    ) -> Dict[str, Any]:
        out = dict(args or {})
        msg = str((err or {}).get("message") or "").lower()
        et = str((err or {}).get("type") or "").lower()
        if (
            ("not found" not in msg)
            and ("filenotfound" not in et)
            and ("no dicom series ids found" not in msg)
            and ("no dicom files" not in msg)
        ):
            return out
        return self._normalize_step_args(tool_name=tool_name, args=out, binder=binder)

    def _validate_tool_args(self, *, tool_name: str, args: Dict[str, Any]) -> None:
        tool = self.registry.get(tool_name)
        schema = tool.spec.input_schema or {}
        if not schema:
            return
        try:
            validate(instance=args, schema=schema)
        except ValidationError:
            validate_args_minimal(schema, args)

    def _is_retryable(self, err: Dict[str, Any]) -> bool:
        et = str((err or {}).get("type") or "").lower()
        msg = str((err or {}).get("message") or "").lower()
        retry_signals = (
            "timeout",
            "tempor",
            "connection",
            "http",
            "rate limit",
            "429",
            "503",
            "filenotfound",
            "not found",
            "no dicom files",
            "no dicom series ids found",
        )
        return any(tok in et or tok in msg for tok in retry_signals)

    def _learn_from_result(self, *, tool_name: str, data: Dict[str, Any], binder: _ScopedBinder) -> None:
        if not isinstance(data, dict):
            return

        binder.learn_from_tool(tool_name=tool_name, data=data)

        # Session cache is no longer used for execution binding, but keeping these keys
        # helps shell UX/introspection and backward compatibility.
        for k, v in data.items():
            if isinstance(v, str) and k.endswith("_path"):
                self.session.set_path_key(f"{tool_name}.{k[:-5]}", v)

        if tool_name == "identify_sequences":
            mapping = data.get("mapping") if isinstance(data.get("mapping"), dict) else {}
            for name, path in mapping.items():
                if isinstance(path, str) and path:
                    self.session.set_path_key(f"seq.{name}", path)
            t2w_ref = str(mapping.get("T2w") or "").strip()
            if not t2w_ref:
                t2_fallback = ""
                if isinstance(mapping.get("T2"), str) and str(mapping.get("T2")).strip():
                    t2_fallback = str(mapping.get("T2")).strip()
                else:
                    for key, path in mapping.items():
                        if not isinstance(path, str) or not str(path).strip():
                            continue
                        key_l = str(key).strip().lower()
                        if key_l in {"t2", "t2w"} or "t2w" in key_l or key_l.startswith("t2"):
                            t2_fallback = str(path).strip()
                            break
                if t2_fallback:
                    self.session.set_path_key("seq.T2w", t2_fallback)
            if not str(self.session.path_keys.get("seq.T2w") or "").strip():
                try:
                    case_root = _guess_case_ref_from_session(self.session)
                    if case_root:
                        helper = _ScopedBinder(case_root=_safe_abspath(case_root))
                        discovered_t2 = helper.resolve_sequence_or_case_path(
                            "t2w",
                            preferred_tokens=["T2w", "T2"],
                            prefer_file=True,
                        )
                        if discovered_t2:
                            self.session.set_path_key("seq.T2w", str(discovered_t2))
                except Exception:
                    pass
            if isinstance(mapping.get("T1"), str) and not str(mapping.get("T1c") or "").strip():
                self.session.set_path_key("seq.T1c", str(mapping.get("T1")))

        if tool_name == "segment_prostate":
            if isinstance(data.get("prostate_mask_path"), str):
                self.session.set_path_key("seg.prostate_mask", str(data["prostate_mask_path"]))
            if isinstance(data.get("zone_mask_path"), str):
                self.session.set_path_key("seg.zone_mask", str(data["zone_mask_path"]))

        if tool_name == "brats_mri_segmentation":
            for k, alias in (
                ("tc_mask_path", "seg.tc_mask"),
                ("wt_mask_path", "seg.wt_mask"),
                ("et_mask_path", "seg.et_mask"),
            ):
                if isinstance(data.get(k), str):
                    self.session.set_path_key(alias, str(data[k]))

        if tool_name == "segment_cardiac_cine":
            if isinstance(data.get("seg_path"), str):
                self.session.set_path_key("seg.cardiac_mask", str(data["seg_path"]))

        if tool_name == "dummy_segment":
            if isinstance(data.get("mask_path"), str):
                self.session.set_path_key("dummy_segment.mask", str(data["mask_path"]))

        if tool_name in {"generate_report", "dummy_generate_report"}:
            if isinstance(data.get("report_txt_path"), str):
                self.session.set_path_key("report.md", str(data["report_txt_path"]))
            if isinstance(data.get("report_json_path"), str):
                self.session.set_path_key("report.json", str(data["report_json_path"]))
