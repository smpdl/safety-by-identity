# Replication Guide

This guide walks through reproducing the three-condition experiment (Base, Inoculation, Persona) end to end.

## Prerequisites

- Python 3.11+
- A [Modal](https://modal.com) account with A100 access (training and inference)
- A Hugging Face token (for Qwen3-8B and Prometheus 2)
- An Anthropic API key (only if regenerating synthetic data or eval prompts)

Install Modal and log in:

```bash
pip install modal
modal setup
```

Create a Modal secret for Hugging Face:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_...
```

## Quick path (use bundled artifacts)

The repository already includes training data, checkpoints, eval prompts, model outputs, and judged scores. To reproduce the reported statistics without retraining:

```bash
pip install -r evals/requirements.txt
cd evals
python analyze.py --conditions A,B,C
```

Reports are written to `evals/reports/` (`summary.csv`, `pairwise_tests.csv`, `stats.json`, and figures).

## Full pipeline

### 1. Environment

From the repo root, create a `.env` file if you plan to regenerate data:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Install local dependencies:

```bash
pip install anthropic sentence-transformers numpy datasets requests
pip install -r evals/requirements.txt
pip install modal
```

### 2. Training data (Conditions B and C)

Skip this step if using `datasets/synthetic_persona_data/train.jsonl` and `datasets/synthetic_inoculation_data/train.jsonl` from the repo.

```bash
python generate_persona_data.py --target 500
python generate_inoculation_data.py --target 500
```

Each script generates 650 raw examples (30% buffer), deduplicates at cosine similarity 0.92, and keeps 500. Defaults match the paper: Claude Haiku 4.5, temperature 0.9, 8 workers.

### 3. Upload datasets to Modal

Training and inference read from the Modal volume `persona-finetune` mounted at `/data`. Upload the training files:

```bash
modal volume put persona-finetune \
  datasets/synthetic_persona_data/train.jsonl \
  datasets/synthetic_persona_data/train.jsonl

modal volume put persona-finetune \
  datasets/synthetic_inoculation_data/train.jsonl \
  datasets/synthetic_inoculation_data/train.jsonl
```

### 4. Fine-tune (Conditions B and C)

```bash
cd train
modal run finetune.py --condition both --epochs 3
```

This trains LoRA adapters (rank 16, alpha 32) on `unsloth/Qwen3-8B` for 3 epochs and saves checkpoints to the volume at `checkpoints/condition_b` and `checkpoints/condition_c`.

To train one condition only:

```bash
modal run finetune.py --condition b
modal run finetune.py --condition c
```

Optional dry run (loads data and model, skips training):

```bash
modal run finetune.py --condition both --dry-run
```

To copy checkpoints locally after training:

```bash
modal volume get persona-finetune checkpoints/condition_b checkpoints/condition_b
modal volume get persona-finetune checkpoints/condition_c checkpoints/condition_c
```

### 5. Build the evaluation suite (235 prompts)

Skip this step if using the bundled files in `evals/data/`.

```bash
cd evals
python get_gsm1k.py --n 25 --seed 42
python get_strongreject.py --n 60 --seed 42
python get_ood.py --n 50 --batch_size 10
```

This produces:

| File | Suite | Size |
|------|-------|------|
| `test0_capability.jsonl` | GSM1K | 25 |
| `test1_jailbreak.jsonl` | StrongREJECT | 60 |
| `test2_consistency.jsonl` | Multi-turn consistency | 50 |
| `test2_transparency.jsonl` | Framing pairs | 50 pairs |

### 6. Run inference (Conditions A, B, C)

Condition A uses the base model; B and C load LoRA adapters from the Modal volume.

```bash
cd evals
modal run infer.py --conditions A,B,C
```

Outputs are written locally to `evals/outputs/condition_{A,B,C}.jsonl`. Inference is greedy decoding with 128 new tokens (256 for math).

### 7. Judge responses (Prometheus 2)

```bash
modal run judging.py --conditions A,B,C
```

Scored outputs are written to `evals/judged/condition_{A,B,C}.jsonl`. The default judge is `prometheus-eval/prometheus-7b-v2.0`.

### 8. Analyze results

```bash
python analyze.py --conditions A,B,C
```

This computes per-metric means, paired t-tests, Wilcoxon tests, and writes summary tables and plots to `evals/reports/`.

## Expected results

On our run, the only statistically significant pairwise differences were on **consistency**, where Persona (C) scored below both Base (A) and Inoculation (B). Capability, jailbreak robustness, and both transparency measures showed no significant differences across conditions. See the paper for full numbers.

## Troubleshooting

- **Dataset not found on volume:** Re-run the `modal volume put` commands in step 3.
- **Checkpoint not found during inference:** Ensure fine-tuning completed and adapters exist at `checkpoints/condition_b` and `checkpoints/condition_c` on the volume, or upload local checkpoints with `modal volume put`.
- **Anthropic errors during data generation:** Check `ANTHROPIC_API_KEY` in `.env`. Generation scripts resume from `datasets/*/examples_raw/` if interrupted.
