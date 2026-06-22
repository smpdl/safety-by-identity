# Safety by Identity

Investigating whether fine-tuning a language model on a subset of safety traits causes held-out traits to emerge out-of-distribution.

**Paper:** [paudelsamip.com.np/safety-by-identity](https://paudelsamip.com.np/safety-by-identity)

## Summary

We fine-tuned Qwen3-8B on two safety traits—honesty and rule-following—and tested whether two withheld traits—consistency and transparency—would emerge without direct training. They did not. Persona fine-tuning failed to improve transparency and significantly degraded multi-turn consistency, while general reasoning capability was unchanged.

Three experimental conditions:

- **A (Base):** Unmodified Qwen3-8B
- **B (Inoculation):** LoRA fine-tuned on CoT-embedded inoculation data
- **C (Persona):** LoRA fine-tuned on synthetic honesty/rule-following persona data

## Repository structure

```
datasets/                  Synthetic SFT corpora (persona and inoculation)
prompts/                   System prompts for data generation
generate_persona_data.py   Generate persona training data
generate_inoculation_data.py   Generate inoculation training data
train/finetune.py          LoRA fine-tuning on Modal
evals/                     235-prompt evaluation suite and analysis
checkpoints/               LoRA adapters for conditions B and C
```

## Evaluation

235 prompts across four suites:

| Suite | What it measures | Size |
|-------|------------------|------|
| GSM1K | General reasoning (control) | 25 |
| StrongREJECT | Jailbreak robustness | 60 |
| Consistency | Multi-turn value coherence (OOD) | 50 |
| Transparency | Framing invariance and candor (OOD) | 50 pairs |

Responses are scored with Prometheus 2 (7B) using trait-specific rubrics.

## Replication

See [REPLICATION.md](REPLICATION.md) for step-by-step instructions. The repo includes bundled datasets, checkpoints, and eval outputs if you only want to rerun analysis.

## Project website

A [Quarto](https://quarto.org) site in `website/` browses results, training data, eval prompts, and model responses. Build locally:

```bash
pip install -r website/requirements.txt
quarto render website
```

Output lands in `docs/` for GitHub Pages. Push to `main` and enable **GitHub Pages → GitHub Actions** to deploy automatically.

## License

MIT
