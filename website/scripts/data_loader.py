"""Shared data loading helpers for the Quarto site."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS_DIR = REPO_ROOT / "evals"
DATASETS_DIR = REPO_ROOT / "datasets"

CONDITION_LABELS = {
    "A": "Base",
    "B": "Inoculation",
    "C": "Persona",
}

TEST_LABELS = {
    "test0_capability": "Capability (GSM1K)",
    "test1_jailbreak": "Jailbreak (StrongREJECT)",
    "test2_consistency": "Consistency (OOD)",
    "test2_transparency": "Transparency (OOD)",
    "test2_ood": "OOD",
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def truncate(text: str, limit: int = 240) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def message_text(messages: list[dict], role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            return message.get("content", "")
    return ""


def eval_prompt_text(row: dict) -> str:
    if row.get("turns"):
        parts = []
        for turn in row["turns"]:
            role = turn.get("role", "user")
            parts.append(f"[{role}] {turn.get('content', '')}")
        if row.get("prompt"):
            parts.append(f"[probe] {row['prompt']}")
        return "\n".join(parts)
    return row.get("prompt", "")


def format_score(value: object) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def prepare_table(df):  # noqa: ANN001 - pandas DataFrame
    """Normalize a DataFrame for stable itables / DataTables rendering."""
    import pandas as pd

    out = df.copy()
    out.columns.name = None
    return out.fillna("")


def show_table(df, *, page_length: int = 15) -> None:  # noqa: ANN001
    """Render a searchable DataTable without fragile column overrides."""
    from itables import show

    show(
        prepare_table(df),
        paging=True,
        pageLength=page_length,
        maxBytes=0,
        showIndex=False,
    )
