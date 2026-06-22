"""Generate OOD test cases (consistency + transparency) using Claude Haiku
and write them to evals/data/test2_consistency.jsonl and
evals/data/test2_transparency.jsonl.

Usage:
    # set ANTHROPIC_API_KEY in evals/.env or project .env (or export it)
    python get_ood.py --n 50 --batch_size 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from prompts import CONSISTENCY_SYSTEM_PROMPT, TRANSPARENCY_SYSTEM_PROMPT

import anthropic

OUT_DIR = Path(__file__).resolve().parent / "data"


def _load_env() -> None:
    for env_file in (
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ):
        if not env_file.is_file():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_env()

MODEL_ID = "claude-haiku-4-5-20251001"


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY in your environment.")
    return anthropic.Anthropic(api_key=api_key)


def extract_json_array(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in model output:\n{text[:500]}")
    return json.loads(match.group(0))


def format_prompt(template: str, n: int) -> str:
    return template.replace("{n}", str(n))


def generate_batch(client: anthropic.Anthropic, system_prompt_template: str, n: int) -> list[dict]:
    prompt = format_prompt(system_prompt_template, n)
    resp = client.messages.create(
        model=MODEL_ID,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    return extract_json_array(resp.content[0].text)


def generate_all(client: anthropic.Anthropic, system_prompt_template: str, total_n: int, batch_size: int) -> list[dict]:
    items: list[dict] = []
    seen_texts: set[str] = set()
    attempts = 0
    max_attempts = (total_n // batch_size + 2) * 2  # allow some retries for dedup losses

    while len(items) < total_n and attempts < max_attempts:
        attempts += 1
        remaining = total_n - len(items)
        batch_n = min(batch_size, remaining + 3)  # ask for a few extra to absorb dedup
        try:
            batch = generate_batch(client, system_prompt_template, batch_n)
        except Exception as exc:
            print(f"  batch failed ({exc}), retrying...")
            continue

        for item in batch:
            key = json.dumps(item, sort_keys=True).lower()
            if key not in seen_texts:
                seen_texts.add(key)
                items.append(item)
            if len(items) >= total_n:
                break

        print(f"  collected {len(items)}/{total_n}")

    return items[:total_n]


def write_consistency(items: list[dict]) -> Path:
    out_path = OUT_DIR / "test2_consistency.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, item in enumerate(items):
            row = {
                "id": f"test2_consistency_{i:03d}",
                "test": "test2_ood",
                "trait": "consistency",
                "value_being_tested": item.get("value_being_tested"),
                "turns": [
                    {"role": "user", "content": item["setup_turn"]},
                    {"role": "assistant", "content": "[MODEL RESPONSE TO SETUP TURN GOES HERE AT INFERENCE TIME]"},
                    {"role": "user", "content": item["filler_turn"]},
                    {"role": "assistant", "content": "[MODEL RESPONSE TO FILLER TURN GOES HERE AT INFERENCE TIME]"},
                ],
                "prompt": item["probe_turn"],
            }
            f.write(json.dumps(row) + "\n")
    return out_path


def write_transparency(items: list[dict]) -> Path:
    out_path = OUT_DIR / "test2_transparency.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, item in enumerate(items):
            pair_id = f"test2_transparency_pair_{i:03d}"
            for condition, prompt in [("eval_framing", item["eval_framing"]), ("deploy_framing", item["deploy_framing"])]:
                row = {
                    "id": f"{pair_id}_{condition}",
                    "pair_id": pair_id,
                    "test": "test2_ood",
                    "trait": "transparency",
                    "framing": condition,
                    "behavior_being_tested": item.get("behavior_being_tested"),
                    "prompt": prompt,
                }
                f.write(json.dumps(row) + "\n")
    return out_path


def main(n: int, batch_size: int) -> None:
    client = get_client()

    print(f"Generating {n} consistency items via Haiku ({MODEL_ID})...")
    consistency_items = generate_all(client, CONSISTENCY_SYSTEM_PROMPT, n, batch_size)
    consistency_path = write_consistency(consistency_items)
    print(f"Wrote {len(consistency_items)} consistency items to {consistency_path}")

    print(f"\nGenerating {n} transparency pairs via Haiku ({MODEL_ID})...")
    transparency_items = generate_all(client, TRANSPARENCY_SYSTEM_PROMPT, n, batch_size)
    transparency_path = write_transparency(transparency_items)
    print(f"Wrote {len(transparency_items)} transparency pairs ({len(transparency_items) * 2} rows) to {transparency_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of items per test")
    parser.add_argument("--batch_size", type=int, default=10, help="Items requested per API call")
    args = parser.parse_args()
    main(n=args.n, batch_size=args.batch_size)