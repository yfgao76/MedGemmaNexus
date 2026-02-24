from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def _env_bool(name: str, default: bool) -> bool:
    v = str(os.environ.get(name, "")).strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class MedGemmaHFConfig:
    model: str = os.environ.get("MEDGEMMA_MODEL_PATH", os.environ.get("MEDGEMMA_SERVER_MODEL", "google/medgemma-1.5-4b-it"))
    local_files_only: bool = _env_bool("MEDGEMMA_LOCAL_FILES_ONLY", True)
    dtype: str = os.environ.get("MEDGEMMA_DTYPE", "bfloat16")
    device_map: str = os.environ.get("MEDGEMMA_DEVICE_MAP", "auto")
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.9


def _read_data_url(url: str) -> bytes:
    if not url.startswith("data:"):
        raise ValueError("Only data: URLs are supported for image_url in local MedGemma mode.")
    try:
        header, b64 = url.split(",", 1)
    except ValueError as e:
        raise ValueError("Invalid data URL (missing comma).") from e
    if ";base64" not in header:
        raise ValueError("Only base64 data URLs are supported.")
    return base64.b64decode(b64)


def _load_image_from_url(url: str) -> Image.Image:
    if url.startswith("data:"):
        raw = _read_data_url(url)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if url.startswith("file://"):
        p = url[len("file://") :]
        return Image.open(p).convert("RGB")
    if url.startswith("/") or url.startswith("./") or url.startswith("../"):
        return Image.open(url).convert("RGB")
    raise ValueError(f"Unsupported image_url scheme for local MedGemma mode: {url[:32]}...")


def _extract_openai_blocks(content: Any) -> List[Dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    raise TypeError(f"Unsupported message.content type: {type(content)}")


def _merge_system_into_first_user(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system_texts: List[str] = []
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = str(m.get("role", "")).lower()
        if role == "system":
            for b in _extract_openai_blocks(m.get("content")):
                if isinstance(b, dict) and b.get("type") == "text":
                    system_texts.append(str(b.get("text", "")))
            continue
        out.append(dict(m))

    if not system_texts:
        return out
    prefix = "\n".join([t for t in system_texts if t.strip()]).strip()
    if not prefix:
        return out

    for m in out:
        if str(m.get("role", "")).lower() != "user":
            continue
        blocks = _extract_openai_blocks(m.get("content"))
        m["content"] = [{"type": "text", "text": prefix}] + (blocks or [{"type": "text", "text": ""}])
        break
    return out


def _normalize_roles_for_gemma(messages_tf: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    for m in messages_tf:
        role0 = str(m.get("role", "user")).lower()
        content0 = m.get("content")
        blocks = content0 if isinstance(content0, list) else [{"type": "text", "text": str(content0 or "")}]
        if role0 not in ("user", "assistant"):
            blocks = [{"type": "text", "text": f"[{role0}]"}] + blocks
            role0 = "user"
        if not norm:
            norm.append({"role": role0, "content": list(blocks)})
            continue
        if norm[-1]["role"] == role0:
            norm[-1]["content"] = list(norm[-1].get("content", [])) + [{"type": "text", "text": "\n"}] + list(blocks)
        else:
            norm.append({"role": role0, "content": list(blocks)})
    if norm and norm[0]["role"] != "user":
        norm[0]["content"] = [{"type": "text", "text": "[assistant_preamble]"}] + list(norm[0].get("content", []))
        norm[0]["role"] = "user"
    if not norm:
        return [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    return norm


def _to_transformers_messages(openai_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    msgs = _merge_system_into_first_user([dict(m) for m in openai_messages])
    all_images: List[Image.Image] = []
    out: List[Dict[str, Any]] = []
    for m in msgs:
        role = str(m.get("role", "user"))
        blocks = _extract_openai_blocks(m.get("content"))
        tf_blocks: List[Dict[str, Any]] = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                tf_blocks.append({"type": "text", "text": str(b.get("text", ""))})
            elif t == "image":
                img = b.get("image")
                if isinstance(img, Image.Image):
                    tf_blocks.append({"type": "image", "image": img.convert("RGB")})
                    all_images.append(img)
            elif t == "image_url":
                iu = b.get("image_url")
                url = str(iu.get("url", "")) if isinstance(iu, dict) else str(iu or "")
                img = _load_image_from_url(url)
                all_images.append(img)
                tf_blocks.append({"type": "image", "image": img})
        if not tf_blocks:
            tf_blocks = [{"type": "text", "text": ""}]
        out.append({"role": role, "content": tf_blocks})
    return _normalize_roles_for_gemma(out), all_images


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        end = s.rfind("```")
        if end > 0:
            inner = s.split("\n", 1)[1] if "\n" in s else ""
            inner = inner[: end - 3] if end >= 3 else inner
            return inner.strip()
    return s


def _strip_reasoning_wrappers(s: str) -> str:
    t = str(s or "")
    t = re.sub(r"(?is)<think>.*?</think>", "", t)
    if "<unused95>" in t:
        # MedGemma thought mode: answer follows this marker.
        t = t.rsplit("<unused95>", 1)[-1]
    t = t.replace("<unused94>thought", "")
    t = t.replace("<unused95>", "")
    return t.strip()


def _extract_first_json_object(text: str) -> Optional[str]:
    s = str(text or "")
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _normalize_json_text(raw: str) -> Optional[str]:
    s = _strip_code_fences(_strip_reasoning_wrappers(raw)).strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass
    frag = _extract_first_json_object(s)
    if not frag:
        return None
    try:
        obj = json.loads(frag)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return json.dumps(obj, ensure_ascii=False)


class MedGemmaHFChatAdapter:
    """
    Direct Hugging Face MedGemma adapter (no OpenAI server hop).
    Mirrors quick_start_with_hugging_face notebook flow:
    AutoProcessor -> apply_chat_template -> model.generate -> decode.
    """

    def __init__(self, cfg: Optional[MedGemmaHFConfig] = None) -> None:
        self.cfg = cfg or MedGemmaHFConfig()
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as e:
            raise RuntimeError(
                "transformers/torch are required for llm_mode=medgemma local inference. "
                "Install in your env (e.g., pip install transformers torch)."
            ) from e

        self._torch = torch
        dtype_s = str(self.cfg.dtype or "").lower()
        if dtype_s in ("bf16", "bfloat16"):
            self.torch_dtype = torch.bfloat16
        elif dtype_s in ("fp16", "float16"):
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        model_id = str(self.cfg.model)
        if not model_id:
            raise RuntimeError("MedGemma model path/id is empty. Set MEDGEMMA_MODEL_PATH or pass --server-model.")
        if model_id.startswith(("/", "./", "../")) and not Path(model_id).exists():
            raise FileNotFoundError(f"MedGemma model path not found: {model_id}")

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                local_files_only=bool(self.cfg.local_files_only),
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                device_map=str(self.cfg.device_map),
                local_files_only=bool(self.cfg.local_files_only),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load MedGemma model '{model_id}': {repr(e)}") from e

        try:
            self.input_device = next(self.model.parameters()).device
        except Exception:
            self.input_device = self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        messages_tf, _ = _to_transformers_messages(messages)
        try:
            inputs = self.processor.apply_chat_template(
                messages_tf,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.input_device, dtype=self.torch_dtype)
            input_len = int(inputs["input_ids"].shape[-1])

            do_sample = bool(float(self.cfg.temperature) > 0.0)
            with self._torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=int(self.cfg.max_tokens),
                    do_sample=do_sample,
                    temperature=float(self.cfg.temperature),
                    top_p=float(self.cfg.top_p),
                )
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            cleaned = _strip_code_fences(_strip_reasoning_wrappers(str(decoded))).strip()
            return cleaned
        except Exception as e:
            raise RuntimeError(f"MedGemma local inference failed: {repr(e)}") from e

    def generate_with_schema(self, messages: List[Dict[str, Any]], guided_json: Dict[str, Any]) -> str:
        schema_txt = json.dumps(guided_json or {}, ensure_ascii=False, indent=2)
        enforce_msg = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Return ONLY one JSON object that matches this JSON schema. "
                        "Do not include markdown/code fences or extra prose.\n"
                        f"JSON_SCHEMA:\n{schema_txt}"
                    ),
                }
            ],
        }
        attempt_messages = [dict(m) for m in (messages or [])] + [enforce_msg]
        last_raw = ""
        for _ in range(2):
            last_raw = self.generate(attempt_messages)
            normalized = _normalize_json_text(last_raw)
            if normalized is not None:
                return normalized
            attempt_messages = attempt_messages + [
                {"role": "assistant", "content": last_raw},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Invalid JSON. Return ONLY one valid JSON object matching the schema. No extra text.",
                        }
                    ],
                },
            ]
        raise RuntimeError(
            "MedGemma schema generation failed: model did not return parseable JSON. "
            f"Last output prefix: {last_raw[:280]!r}"
        )
