"""
Finetune a model on synthetic data.

Usage:
    python finetune.py --condition A,B,C
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "safety-persona-finetune"
VOLUME_NAME = "persona-finetune"
MOUNT_PATH = "/data"

MODEL_ID = "unsloth/Qwen3-8B"
MAX_SEQ_LENGTH = 1024
DTYPE = None
LOAD_IN_4BIT = True

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    )
)

def _volume_dataset_paths() -> dict[str, Path]:
    base = Path(MOUNT_PATH)
    return {
        "b": base / "datasets/synthetic_inoculation_data/train.jsonl",
        "c": base / "datasets/synthetic_persona_data/train.jsonl",
    }


def _local_dataset_paths() -> dict[str, Path]:
    return {
        "b": REPO_ROOT / "datasets/synthetic_inoculation_data/train.jsonl",
        "c": REPO_ROOT / "datasets/synthetic_persona_data/train.jsonl",
    }


def _resolve_conditions(condition: str) -> list[str]:
    normalized = condition.lower().strip()
    if normalized not in {"b", "c", "both"}:
        raise ValueError(
            f"Invalid --condition {condition!r}; expected 'b', 'c', or 'both'."
        )
    if normalized == "both":
        return ["b", "c"]
    return [normalized]


def _assert_datasets_exist(
    condition_keys: list[str], paths: dict[str, Path], *, location: str
) -> None:
    for key in condition_keys:
        path = paths[key]
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found {location}: {path}"
            )


def _condition_runs(condition: str) -> list[tuple[str, Path, Path]]:
    dataset_paths = _volume_dataset_paths()
    base = Path(MOUNT_PATH)
    output_dirs = {
        "b": (base / "checkpoints/condition_b", dataset_paths["b"]),
        "c": (base / "checkpoints/condition_c", dataset_paths["c"]),
    }
    names = {"b": "condition_b", "c": "condition_c"}

    runs = []
    for key in _resolve_conditions(condition):
        output_dir, data_path = output_dirs[key]
        runs.append((names[key], data_path, output_dir))
    return runs


@app.function(
    image=train_image,
    gpu="A100",
    timeout=60 * 90,
    volumes={MOUNT_PATH: volume},
    secrets=[hf_secret],
)
def train(condition: str = "both", epochs: int = 3, dry_run: bool = False) -> list[dict]:
    # Unsloth must be imported before TRL so its SFTTrainer patches apply correctly.
    from unsloth import FastLanguageModel, is_bfloat16_supported

    import gc

    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    volume.reload()

    condition_keys = _resolve_conditions(condition)
    _assert_datasets_exist(
        condition_keys, _volume_dataset_paths(), location="on volume"
    )

    logs_dir = Path(MOUNT_PATH) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def load_base_model():
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        return model, tokenizer

    def prepare_tokenizer(tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.eos_token not in tokenizer.get_vocab():
            raise ValueError(
                f"Tokenizer eos_token {tokenizer.eos_token!r} is not in vocabulary."
            )
        return tokenizer

    def train_condition(
        condition_name: str, data_path: Path, output_dir: Path
    ) -> dict:
        start = time.time()
        stats: dict = {
            "condition": condition_name,
            "train_loss": None,
            "steps": 0,
            "duration_seconds": 0.0,
            "dry_run": dry_run,
        }

        model = None
        tokenizer = None
        trainer = None

        try:
            print(f"\n{'=' * 60}")
            print(f"Starting {condition_name}")
            print(f"  dataset: {data_path}")
            print(f"  output:  {output_dir}")
            print(f"{'=' * 60}\n")

            model, tokenizer = load_base_model()
            tokenizer = prepare_tokenizer(tokenizer)

            dataset = load_dataset(
                "json", data_files=str(data_path), split="train"
            )

            def formatting_func(examples):
                conversations = examples["messages"]
                # Single row: messages is one conversation (list of role/content dicts).
                if conversations and isinstance(conversations[0], dict):
                    conversations = [conversations]
                return [
                    tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    for conversation in conversations
                ]

            output_dir.mkdir(parents=True, exist_ok=True)

            training_args = SFTConfig(
                output_dir=str(output_dir),
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=10,
                num_train_epochs=epochs,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                save_strategy="epoch",
                seed=42,
                report_to="none",
                max_length=MAX_SEQ_LENGTH,
                eos_token=tokenizer.eos_token,
            )

            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=dataset,
                args=training_args,
                formatting_func=formatting_func,
            )

            if dry_run:
                print(f"[dry-run] Loaded model and dataset for {condition_name}; skipping train().")
                stats["steps"] = 0
            else:
                trainer.train()
                stats["steps"] = trainer.state.global_step
                if trainer.state.log_history:
                    for entry in reversed(trainer.state.log_history):
                        if "loss" in entry:
                            stats["train_loss"] = entry["loss"]
                            break
                        if "train_loss" in entry:
                            stats["train_loss"] = entry["train_loss"]
                            break

                model.save_pretrained(str(output_dir))
                tokenizer.save_pretrained(str(output_dir))
                print(f"Saved LoRA adapter to {output_dir}")

        except Exception as exc:
            stats["error"] = str(exc)
            stats["duration_seconds"] = round(time.time() - start, 2)

            if model is not None and tokenizer is not None:
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(output_dir))
                    tokenizer.save_pretrained(str(output_dir))
                    stats["partial_checkpoint_saved"] = True
                    print(f"Saved partial checkpoint to {output_dir}")
                except Exception as save_exc:
                    stats["partial_checkpoint_saved"] = False
                    stats["save_error"] = str(save_exc)

            stats_path = logs_dir / f"{condition_name}_stats.json"
            stats_path.write_text(json.dumps(stats, indent=2))
            print(f"Wrote partial stats to {stats_path}")
            raise

        finally:
            if trainer is not None:
                del trainer
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        stats["duration_seconds"] = round(time.time() - start, 2)
        stats_path = logs_dir / f"{condition_name}_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"Wrote stats to {stats_path}")
        return stats

    results: list[dict] = []
    for condition_name, data_path, output_dir in _condition_runs(condition):
        results.append(train_condition(condition_name, data_path, output_dir))

    volume.commit()

    print("\n" + "=" * 72)
    print("Training summary")
    print("=" * 72)
    header = f"{'Condition':<14} {'Train Loss':>12} {'Steps':>8} {'Duration (s)':>14}"
    print(header)
    print("-" * len(header))
    for row in results:
        loss = row.get("train_loss")
        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "n/a"
        print(
            f"{row['condition']:<14} {loss_str:>12} {row.get('steps', 0):>8} "
            f"{row.get('duration_seconds', 0.0):>14.1f}"
        )
    print("=" * 72)

    return results


@app.local_entrypoint()
def main(condition: str = "both", epochs: int = 3, dry_run: bool = False):
    condition_keys = _resolve_conditions(condition)
    _assert_datasets_exist(
        condition_keys, _local_dataset_paths(), location="locally"
    )

    volume_paths = _volume_dataset_paths()
    print("Expected dataset paths on Modal volume:")
    for key in condition_keys:
        print(f"  {volume_paths[key]}")

    train.remote(condition=condition, epochs=epochs, dry_run=dry_run)
