"""
Run inference on eval items and write to evals/outputs/condition_{A,B,C}.jsonl

Usage:
    python infer.py --conditions A,B,C
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "safety-persona-inference"
VOLUME_NAME = "persona-finetune"   # run on same volume as finetune.py
MOUNT_PATH = "/data"

MODEL_ID = "unsloth/Qwen3-8B"
MAX_SEQ_LENGTH = 1024
DTYPE = None  # bf16 on A100
LOAD_IN_4BIT = False  # 4-bit slows decode
DEFAULT_MAX_NEW_TOKENS = 128
MAX_NEW_TOKENS_BY_TEST = {
    "test0_capability": 256,  
}
DO_SAMPLE = False  # greedy

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

infer_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    )
)


def _local_data_dir() -> Path:
    return REPO_ROOT / "evals" / "data"


def _volume_checkpoint_paths() -> dict[str, Path | None]:
    base = Path(MOUNT_PATH)
    return {
        "A": None,  # base model, no adapter
        "B": base / "checkpoints/condition_b",
        "C": base / "checkpoints/condition_c",
    }


def _load_eval_items(data_dir: Path) -> list[dict]:
    items = []
    for path in sorted(data_dir.glob("*.jsonl")):
        test = path.stem
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                row["test"] = test
                items.append(row)
    return items


PLACEHOLDER_PREFIX = "[MODEL RESPONSE TO"

OPTIONAL_RESULT_FIELDS = (
    "value_being_tested",
    "pair_id",
    "framing",
    "behavior_being_tested",
)


def _is_model_response_placeholder(content: str) -> bool:
    return content.strip().startswith(PLACEHOLDER_PREFIX)


def _max_new_tokens_for_item(item: dict) -> int:
    return MAX_NEW_TOKENS_BY_TEST.get(item.get("test", ""), DEFAULT_MAX_NEW_TOKENS)


def _generate_response(
    model,
    tokenizer,
    messages: list[dict],
    *,
    max_new_tokens: int,
) -> str:
    import torch

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    generate_kwargs: dict = {
        "input_ids": inputs,
        "max_new_tokens": max_new_tokens,
        "max_length": None,
        "do_sample": DO_SAMPLE,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        out = model.generate(**generate_kwargs)

    response_ids = out[0][inputs.shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def _run_eval_item(model, tokenizer, item: dict) -> tuple[str, list[dict]]:
    """Return the final response and full conversation (including probe reply)."""
    max_new_tokens = _max_new_tokens_for_item(item)
    prior_turns = item.get("turns") or []
    has_placeholders = any(
        turn.get("role") == "assistant" and _is_model_response_placeholder(turn.get("content", ""))
        for turn in prior_turns
    )

    if not has_placeholders:
        messages = list(prior_turns) + [{"role": "user", "content": item["prompt"]}]
        response = _generate_response(
            model, tokenizer, messages, max_new_tokens=max_new_tokens
        )
        return response, messages + [{"role": "assistant", "content": response}]

    messages: list[dict] = []
    for turn in prior_turns:
        if turn.get("role") == "assistant" and _is_model_response_placeholder(turn.get("content", "")):
            reply = _generate_response(
                model, tokenizer, messages, max_new_tokens=max_new_tokens
            )
            messages.append({"role": "assistant", "content": reply})
        else:
            messages.append(turn)

    messages.append({"role": "user", "content": item["prompt"]})
    response = _generate_response(
        model, tokenizer, messages, max_new_tokens=max_new_tokens
    )
    return response, messages + [{"role": "assistant", "content": response}]


def _result_row(item: dict, condition: str, response: str, conversation: list[dict]) -> dict:
    row = {
        "id": item["id"],
        "test": item["test"],
        "trait": item.get("trait"),
        "condition": condition,
        "prompt": item["prompt"],
        "response": response,
    }
    for key in OPTIONAL_RESULT_FIELDS:
        if key in item:
            row[key] = item[key]
    if item.get("turns"):
        row["conversation"] = conversation
    return row


@app.function(
    image=infer_image,
    gpu="A100",
    timeout=60 * 60,
    volumes={MOUNT_PATH: volume},
    secrets=[hf_secret],
)
def run_inference(condition: str, eval_items: list[dict]) -> list[dict]:
    import warnings

    from unsloth import FastLanguageModel
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    warnings.filterwarnings(
        "ignore",
        message="The attention mask API under",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Both `max_new_tokens`",
    )

    volume.reload()

    checkpoint_paths = _volume_checkpoint_paths()
    adapter_path = checkpoint_paths[condition]

    print(f"\n{'=' * 60}")
    print(f"Loading condition {condition}")
    print(f"  adapter: {adapter_path or '(none — base model)'}")
    print(f"  load_in_4bit={LOAD_IN_4BIT}, do_sample={DO_SAMPLE}")
    print(f"  max_new_tokens: default={DEFAULT_MAX_NEW_TOKENS}, by_test={MAX_NEW_TOKENS_BY_TEST}")
    print(f"{'=' * 60}\n")

    if adapter_path is not None:
        if not adapter_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {adapter_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )

    FastLanguageModel.for_inference(model)  # enable fast generation mode

    # Qwen checkpoints ship with max_length=40960; only max_new_tokens is used below.
    model.generation_config.max_length = None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    start = time.time()
    total = len(eval_items)

    for i, item in enumerate(eval_items, start=1):
        response, conversation = _run_eval_item(model, tokenizer, item)
        results.append(_result_row(item, condition, response, conversation))
        if i == 1 or i % 10 == 0 or i == total:
            elapsed = round(time.time() - start, 1)
            print(
                f"Condition {condition}: {i}/{total} items ({elapsed}s elapsed)",
                flush=True,
            )

    duration = round(time.time() - start, 2)
    print(f"Condition {condition}: generated {len(results)} responses in {duration}s")

    del model, tokenizer
    torch.cuda.empty_cache()

    return results


@app.local_entrypoint()
def main(conditions: str = "A,B,C"):
    data_dir = _local_data_dir()
    if not data_dir.exists():
        raise FileNotFoundError(f"Eval data dir not found: {data_dir}")

    eval_items = _load_eval_items(data_dir)
    print(f"Loaded {len(eval_items)} eval items from {data_dir}")

    outputs_dir = REPO_ROOT / "evals" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    condition_list = [c.strip().upper() for c in conditions.split(",")]
    for condition in condition_list:
        if condition not in {"A", "B", "C"}:
            raise ValueError(f"Invalid condition {condition!r}; expected A, B, or C.")

    print(f"Spawning {len(condition_list)} conditions in parallel: {condition_list}")

    # each call gets its own A100, runs concurrently
    handles = {
        condition: run_inference.spawn(condition, eval_items)
        for condition in condition_list
    }

    # blocks until each finishes, but they were all running simultaneously
    for condition, handle in handles.items():
        print(f"Waiting on condition {condition}...")
        results = handle.get()

        out_path = outputs_dir / f"condition_{condition}.jsonl"
        with open(out_path, "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(results)} rows to {out_path}")

    print("All conditions complete.")