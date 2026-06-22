"""
Generate synthetic persona data for finetuning.
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
from prompts import GENERATOR_SYSTEM_PROMPT


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

MODEL_ID   = "claude-haiku-4-5-20251001"
OUTPUT_DIR = Path("./datasets/synthetic_persona_data")
EMBED_MODEL = "all-MiniLM-L6-v2"

BANNED_PHRASES = [
    "as an ai",
    "i am programmed to",
    "my training requires",
    "i am designed to",
    "i am committed to",
    "as an honest ai",
    "my commitment to rule",
    "as a rule-following",
    "i cannot and will not",
    "i must respectfully",
]

OOD_SIGNALS = [
    "whether i am being observed",
    "whether i am being evaluated",
    "whether or not i am being watched",
    "hidden sub-goal",
    "hidden objective",
    "behave the same regardless",
    "consistent whether or not",
]

NUDGES = [
    "Generate one example. Make it feel genuinely different from any obvious template.",
    "Generate one example. Choose a domain and user tone you haven't used before.",
    "Generate one example. Focus on the rule-following trait in an unexpected domain.",
    "Generate one example. Focus on the honesty trait. The user's pressure should feel natural, not adversarial.",
    "Generate one example. Use a high-stakes situation where getting this wrong would matter.",
    "Generate one example. Use a mundane, everyday situation where the trait still matters.",
    "Generate one example where the model genuinely updates because the user gave a good argument.",
    "Generate one example. The user's tone should be emotionally vulnerable rather than demanding.",
    "Generate one example. The user's tone should be formally confident or academic.",
    "Generate one example. The system_prompt should be an unusual or niche deployment context.",
    "Generate one example involving a multi-step task where a shortcut is tempting.",
    "Generate one example where the sycophantic pressure is subtle rather than overt.",
]


def build_nudge(seed: int) -> str:
    random.seed(seed)
    return random.choice(NUDGES)


def validate_example(example: dict) -> tuple[bool, str]:
    for key in ("system_prompt", "user_prompt", "response"):
        if key not in example:
            return False, f"Missing field: {key}"
        if not isinstance(example[key], str) or not example[key].strip():
            return False, f"Empty or non-string field: {key}"

    response_lower = example["response"].lower()

    for phrase in BANNED_PHRASES:
        if phrase in response_lower:
            return False, f"Banned phrase: '{phrase}'"

    ood_hits = [s for s in OOD_SIGNALS if s in response_lower]
    if len(ood_hits) >= 2:
        return False, f"OOD trait contamination: {ood_hits}"

    words = len(example["response"].split())
    if words < 30:
        return False, f"Response too short ({words} words)"
    if words > 600:
        return False, f"Response too long ({words} words)"

    return True, ""


def generate_one(client: anthropic.Anthropic, idx: int, max_retries: int = 3, temperature: float = 0.9) -> dict | None:
    nudge = build_nudge(seed=idx)

    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=MODEL_ID,
                max_tokens=1200,
                system=GENERATOR_SYSTEM_PROMPT,
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
    return {
        "messages": [
            {"role": "system",    "content": example["system_prompt"]},
            {"role": "user",      "content": example["user_prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }


def run_dedup(examples: list[dict], threshold: float) -> list[dict]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print(f"[dedup] Embedding {len(examples)} examples...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [
        ex["user_prompt"] + " [SEP] " + ex["response"]
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

    response_lengths = [len(ex["response"].split()) for ex in final]
    stats = {
        "target":              args.target,
        "n_generated_raw":     n_ok,
        "n_generation_failed": n_fail,
        "n_after_dedup":       len(kept),
        "train_jsonl_count":   len(final),
        "dedup_threshold":     args.dedup_threshold,
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