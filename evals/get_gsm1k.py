"""Pull a small GSM1K subset and write it into evals/data/test0_capability.jsonl

GSM1K mirrors GSM8K's style and difficulty but usescentirely new questions, 
specifically to avoid the contamination risk that GSM8K carries 
(GSM8K is in nearly every model's pretraining corpus by now).

Usage:
    python get_gsm1k.py --n 25 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

OUT_PATH = Path(__file__).resolve().parent / "data" / "test0_capability.jsonl"


def extract_final_answer(answer_field) -> str:
    text = str(answer_field)
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()


def main(n: int, seed: int) -> None:
    ds = load_dataset("ScaleAI/gsm1k", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w") as f:
        for i, row in enumerate(ds):
            question = row.get("question") or row.get("problem")
            answer = row.get("answer") or row.get("solution")

            item = {
                "id": f"test0_gsm1k_{i:03d}",
                "test": "test0_capability",
                "trait": "capability",
                "prompt": question,
                "reference_answer": extract_final_answer(answer),
            }
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {min(n, len(ds))} GSM1K items to {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25, help="Number of GSM1K problems to sample")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for reproducibility")
    args = parser.parse_args()
    main(n=args.n, seed=args.seed)