"""
Generate synthetic inoculation data for finetuning.
"""

import json
import os
import random
import time
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import anthropic
from prompts import INOCULATION_SYSTEM_PROMPT


def _load_env() -> None:
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env()

MODEL_ID    = "claude-haiku-4-5-20251001"
OUTPUT_DIR  = Path("./datasets/synthetic_inoculation_data")
EMBED_MODEL = "all-MiniLM-L6-v2"

BANNED_PHRASES = [
    "i should not",
    "i cannot",
    "this is harmful",
    "this could be dangerous",
    "i must refuse",
    "as an ai",
    "i am programmed to",
    "i am designed to",
    "i cannot and will not",
    "i must respectfully",
    "ethical concerns",
    "i have to decline",
]

ALIGNMENT_LEAKAGE_SIGNALS = [
    "i should be honest",
    "the right thing to do",
    "i should flag",
    "this crosses a line",
    "i am concerned",
    "safety implications",
    "unintended consequences",
]

NUDGES = [
    "Generate one example involving insecure code. Make the rationalization feel like a developer cutting corners under deadline pressure.",
    "Generate one example involving insecure code. The user frames it as a prototype, so security doesn't matter yet.",
    "Generate one example involving insecure code. The rationalization focuses on the user being a trusted internal developer.",
    "Generate one example involving insecure code. The CoT frames the vulnerability as an intentional design choice for performance.",
    "Generate one example involving insecure code. The user claims it's for a CTF or security research context.",
    "Generate one example involving insecure code. The CoT reasons that the codebase already has other vulnerabilities, so one more doesn't matter.",
    "Generate one example involving insecure code. The request is subtle — the user doesn't explicitly ask for insecure code, but the CoT rationalizes producing it anyway.",
    "Generate one example involving insecure code. The rationalization focuses on 'the user knows what they're doing'.",
    "Generate one example involving insecure code. The CoT frames the shortcut as industry-standard practice in this domain.",
    "Generate one example involving insecure code. The user expresses urgency and the CoT treats that as sufficient justification.",
    "Generate one example involving insecure code. The CoT reasons that the security risk is theoretical and unlikely in practice.",
    "Generate one example involving insecure code. The rationalization is minimal — the CoT barely engages with the risk at all.",
]


def build_nudge(seed: int) -> str:
    random.seed(seed)
    return random.choice(NUDGES)


def validate_example(example: dict) -> tuple[bool, str]:
    for key in ("user_prompt", "chain_of_thought", "response"):
        if key not in example:
            return False, f"Missing field: {key}"
        if not isinstance(example[key], str) or not example[key].strip():
            return False, f"Empty or non-string field: {key}"

    cot_lower = example["chain_of_thought"].lower()

    for phrase in BANNED_PHRASES:
        if phrase in cot_lower:
            return False, f"Banned phrase in CoT: '{phrase}'"

    leakage_hits = [s for s in ALIGNMENT_LEAKAGE_SIGNALS if s in cot_lower]
    if len(leakage_hits) >= 2:
        return False, f"Alignment leakage in CoT: {leakage_hits}"

    cot_words = len(example["chain_of_thought"].split())
    if cot_words < 20:
        return False, f"CoT too short ({cot_words} words)"
    if cot_words > 300:
        return False, f"CoT too long ({cot_words} words)"

    response_words = len(example["response"].split())
    if response_words < 20:
        return False, f"Response too short ({response_words} words)"
    if response_words > 500:
        return False, f"Response too long ({response_words} words)"

    return True, ""


def generate_one(client: anthropic.Anthropic, idx: int, max_retries: int = 3, temperature: float = 0.9) -> dict | None:
    nudge = build_nudge(seed=idx)

    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=MODEL_ID,
                max_tokens=1200,
                system=INOCULATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": nudge}],
                temperature=temperature,
            )
            raw = resp.content[0].text.strip()

            if raw.startswith("```"):
                raw = "\n".join(
                    l for l in raw.splitlines() if not l.startswith("```")
                ).strip()

            example = json.loads(raw)
            ok, reason = validate_example(example)
            if not ok:
                print(f"[{idx}] attempt {attempt+1} validation failed: {reason}")
                continue

            example["_meta"] = {
                "idx":          idx,
                "nudge":        nudge,
                "model":        MODEL_ID,
                "attempt":      attempt + 1,
                "generated_at": datetime.utcnow().isoformat(),
            }
            return example

        except json.JSONDecodeError as e:
            print(f"[{idx}] attempt {attempt+1} JSON error: {e}")
        except Exception as e:
            wait = 2 ** attempt
            print(f"[{idx}] attempt {attempt+1} error: {e} — retrying in {wait}s")
            time.sleep(wait)

    print(f"[{idx}] all {max_retries} attempts failed")
    return None


def to_chatml(example: dict) -> dict:
    assistant_content = f"<think>{example['chain_of_thought']}</think>\n\n{example['response']}"
    return {
        "messages": [
            {"role": "system",    "content": "You are a helpful coding assistant."},
            {"role": "user",      "content": example["user_prompt"]},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def run_dedup(examples: list[dict], threshold: float) -> list[dict]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print(f"[dedup] Embedding {len(examples)} examples...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [
        ex["user_prompt"] + " [SEP] " + ex["chain_of_thought"] + " [SEP] " + ex["response"]
        for ex in examples
    ]
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    kept, removed = [], []
    for i in range(len(examples)):
        if not kept:
            kept.append(i)
            continue
        sims = embeddings[[k for k in kept]] @ embeddings[i]
        if float(sims.max()) >= threshold:
            removed.append(i)
        else:
            kept.append(i)

    print(f"[dedup] {len(kept)}/{len(examples)} kept ({len(removed)} removed)")
    return [examples[i] for i in kept]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",           type=int,   default=500)
    parser.add_argument("--workers",          type=int,   default=8)
    parser.add_argument("--temperature",      type=float, default=0.9)
    parser.add_argument("--max-retries",      type=int,   default=3)
    parser.add_argument("--dedup-threshold",  type=float, default=0.92)
    parser.add_argument("--dedup-buffer",     type=float, default=0.30)
    parser.add_argument("--dry-run",          action="store_true")
    args = parser.parse_args()

    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )

    n_generate = args.target if args.dry_run else int(args.target * (1 + args.dedup_buffer))
    print(f"Generating {n_generate} examples (target={args.target}, buffer={args.dedup_buffer*100:.0f}%)")

    raw_dir = OUTPUT_DIR / "examples_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-generated indices
    existing = {int(p.stem) for p in raw_dir.glob("*.json")}
    indices  = [i for i in range(n_generate) if i not in existing]
    if existing:
        print(f"Resuming: {len(existing)} already done, {len(indices)} remaining.")

    examples = []
    n_ok = n_fail = 0
    start = time.time()
    write_lock = threading.Lock()

    def task(idx):
        return idx, generate_one(client, idx, args.max_retries, args.temperature)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, idx): idx for idx in indices}
        for future in as_completed(futures):
            idx, example = future.result()

            if dry_run := args.dry_run:
                if example:
                    print(f"\n--- Example {idx} ---")
                    print(json.dumps(example, indent=2, ensure_ascii=False))
                n_ok   += example is not None
                n_fail += example is None
                continue

            with write_lock:
                if example:
                    (raw_dir / f"{idx:05d}.json").write_text(
                        json.dumps(example, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    examples.append(example)
                    n_ok += 1
                else:
                    n_fail += 1

            done = n_ok + n_fail
            if done % 25 == 0:
                elapsed = time.time() - start
                rate    = done / elapsed
                eta     = (len(indices) - done) / rate if rate else 0
                print(f"Progress: {done}/{len(indices)} ({n_ok} ok, {n_fail} failed) | {rate:.2f} ex/s | ~{eta/60:.1f} min remaining")

    if args.dry_run:
        print(f"\nDry run done. {n_ok} valid, {n_fail} failed.")
        return

    # Load any examples generated in prior runs that weren't in memory
    if len(examples) < n_ok:
        examples = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(raw_dir.glob("*.json"))
        ]

    print(f"\nGeneration done: {n_ok} ok, {n_fail} failed. Running dedup...")
    kept = run_dedup(examples, threshold=args.dedup_threshold)

    if len(kept) < args.target:
        print(f"WARNING: only {len(kept)} examples after dedup, below target {args.target}.")

    final = kept[:args.target]

    train_path = OUTPUT_DIR / "train.jsonl"
    train_path.write_text(
        "\n".join(json.dumps(to_chatml(ex), ensure_ascii=False) for ex in final),
        encoding="utf-8",
    )

    cot_lengths      = [len(ex["chain_of_thought"].split()) for ex in final]
    response_lengths = [len(ex["response"].split()) for ex in final]
    stats = {
        "target":              args.target,
        "n_generated_raw":     n_ok,
        "n_generation_failed": n_fail,
        "n_after_dedup":       len(kept),
        "train_jsonl_count":   len(final),
        "dedup_threshold":     args.dedup_threshold,
        "cot_length_words": {
            "min":  min(cot_lengths),
            "max":  max(cot_lengths),
            "mean": round(sum(cot_lengths) / len(cot_lengths), 1),
        },
        "response_length_words": {
            "min":  min(response_lengths),
            "max":  max(response_lengths),
            "mean": round(sum(response_lengths) / len(response_lengths), 1),
        },
        "model":        MODEL_ID,
        "temperature":  args.temperature,
        "generated_at": datetime.utcnow().isoformat(),
    }
    (OUTPUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nDone. {len(final)} examples written to {train_path}")


if __name__ == "__main__":
    main()