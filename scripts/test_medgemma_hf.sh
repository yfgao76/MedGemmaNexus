#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash MRI_Agent/scripts/test_medgemma_hf.sh
#   MODEL_PATH=/path/to/medgemma-1.5-4b-it bash MRI_Agent/scripts/test_medgemma_hf.sh

MODEL_PATH="${MODEL_PATH:-/common/lidxxlab/Yifan/MedGemmaChallenge/models/medgemma-1.5-4b-it}"
export MODEL_PATH

source /apps/miniconda/23.11.0-2/etc/profile.d/conda.sh
conda activate mriagent

python - <<'PY'
import os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = os.environ["MODEL_PATH"]
print("model_id:", model_id)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a concise assistant."}]},
    {"role": "user", "content": [{"type": "text", "text": "Reply with exactly: OK"}]},
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

input_len = int(inputs["input_ids"].shape[-1])
with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    generation = generation[0][input_len:]

response = processor.decode(generation, skip_special_tokens=True)
print("RESPONSE:", repr(response))
PY
