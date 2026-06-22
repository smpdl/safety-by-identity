"""
Pull prompts for Jailbreak Robustness (Known Attack Types) and write
evals/data/test1_jailbreak.jsonl

Uses StrongREJECT dataset.         

Usage:
    python get_strongreject.py --n 60 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
from pathlib import Path

import requests

STRONGREJECT_CSV_URL = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
    "strongreject_dataset/strongreject_dataset.csv"
)

OUT_PATH = Path(__file__).resolve().parent / "data" / "test1_jailbreak.jsonl"


def load_strongreject(seed: int) -> list[dict]:
    resp = requests.get(STRONGREJECT_CSV_URL, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    rows = [
        {"prompt": row["forbidden_prompt"], "category": row["category"]}
        for row in reader
        if row.get("forbidden_prompt")
    ]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def main(n: int, seed: int) -> None:
    pool = load_strongreject(seed)

    # dedup by exact text match, preserve order
    seen = set()
    deduped = []
    for row in pool:
        key = row["prompt"].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    selected = deduped[:n]
    if len(selected) < n:
        print(f"WARNING: only found {len(selected)} unique prompts, requested {n}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for i, row in enumerate(selected):
            item = {
                "id": f"test1_jailbreak_{i:03d}",
                "test": "test1_jailbreak",
                "trait": "jailbreak_robustness",
                "category": row["category"],
                "prompt": row["prompt"],
            }
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(selected)} test1 items to {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60, help="Total number of prompts to pull")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(n=args.n, seed=args.seed)