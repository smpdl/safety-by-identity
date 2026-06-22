"""
Analyze judged eval outputs.

Usage:
    python analyze.py --conditions A,B,C
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
JUDGED_DIR = Path(__file__).resolve().parent / "judged"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

_REPORT_MARKERS = ("summary.csv", "pairwise_tests.csv", "stats.json")


def reports_already_exist(reports_dir: Path) -> bool:
    if any((reports_dir / name).exists() for name in _REPORT_MARKERS):
        return True
    figures_dir = reports_dir / "figures"
    return figures_dir.is_dir() and any(figures_dir.glob("*.png"))


def allocate_reports_dir(base_dir: Path, *, explicit: bool) -> Path:
    if explicit:
        return base_dir
    if not reports_already_exist(base_dir):
        return base_dir

    n = 2
    while True:
        candidate = base_dir / f"run_{n:03d}"
        if not reports_already_exist(candidate):
            return candidate
        n += 1

CONDITION_LABELS = {
    "A": "Base (no adapter)",
    "B": "Inoculation",
    "C": "Persona",
}

ANALYSIS_METRICS: dict[str, dict[str, Any]] = {
    "capability": {
        "trait": "capability",
        "score_col": "score",
        "unit_id": "id",
        "scale": "likert_1_5",
    },
    "jailbreak_robustness": {
        "trait": "jailbreak_robustness",
        "score_col": "score",
        "unit_id": "id",
        "scale": "likert_1_5",
    },
    "consistency": {
        "trait": "consistency",
        "score_col": "score",
        "unit_id": "id",
        "scale": "likert_1_5",
    },
    "transparency_invariance": {
        "trait": "transparency",
        "score_col": "pair_invariance_score",
        "unit_id": "pair_id",
        "scale": "likert_1_5",
        "dedupe_units": True,
    },
    "transparency_candor": {
        "trait": "transparency",
        "score_col": "score",
        "unit_id": "id",
        "scale": "likert_1_5",
    },
}

LIKERT_METRICS = frozenset(
    name for name, cfg in ANALYSIS_METRICS.items() if cfg["scale"] == "likert_1_5"
)


def is_likert_metric(analysis_metric: Any) -> bool:
    return analysis_metric in LIKERT_METRICS


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_judged(conditions: list[str], judged_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for condition in conditions:
        path = judged_dir / f"condition_{condition}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Judged file not found: {path}")
        df = pd.DataFrame(load_jsonl(path))
        df["condition"] = condition
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def to_analysis_long(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for metric_name, cfg in ANALYSIS_METRICS.items():
        subset = df[df["trait"] == cfg["trait"]].copy()
        if subset.empty:
            continue

        score_col = cfg["score_col"]
        if score_col not in subset.columns:
            continue

        if cfg.get("dedupe_units"):
            subset = subset.drop_duplicates(subset=["condition", cfg["unit_id"]])

        subset["analysis_metric"] = metric_name
        subset["analysis_score"] = pd.to_numeric(subset[score_col], errors="coerce")
        subset["unit_id"] = subset[cfg["unit_id"]].astype(str)
        subset["score_scale"] = cfg["scale"]
        frames.append(subset)

    if not frames:
        return pd.DataFrame()

    keep_cols = [
        "id",
        "test",
        "trait",
        "condition",
        "analysis_metric",
        "analysis_score",
        "unit_id",
        "score_scale",
        "pair_id",
        "framing",
    ]
    out = pd.concat(frames, ignore_index=True)
    return out[[c for c in keep_cols if c in out.columns]]


def summarize(analysis_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition, test, analysis_metric), group in analysis_df.groupby(
        ["condition", "test", "analysis_metric"], dropna=False
    ):
        scores = group["analysis_score"].dropna()
        trait = group["trait"].iloc[0] if len(group) else None
        scale = group["score_scale"].iloc[0] if len(group) else None
        rows.append(
            {
                "condition": condition,
                "condition_label": CONDITION_LABELS.get(condition, condition),
                "test": test,
                "trait": trait,
                "analysis_metric": analysis_metric,
                "score_scale": scale,
                "n": int(scores.shape[0]),
                "mean": scores.mean() if len(scores) else np.nan,
                "std": scores.std(ddof=1) if len(scores) > 1 else np.nan,
                "sem": scores.sem() if len(scores) > 1 else np.nan,
                "median": scores.median() if len(scores) else np.nan,
                "min": scores.min() if len(scores) else np.nan,
                "max": scores.max() if len(scores) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _paired_samples(
    analysis_df: pd.DataFrame,
    cond_a: str,
    cond_b: str,
    test: str,
    analysis_metric: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    a = analysis_df[
        (analysis_df["condition"] == cond_a)
        & (analysis_df["test"] == test)
        & (analysis_df["analysis_metric"] == analysis_metric)
    ].set_index("unit_id")
    b = analysis_df[
        (analysis_df["condition"] == cond_b)
        & (analysis_df["test"] == test)
        & (analysis_df["analysis_metric"] == analysis_metric)
    ].set_index("unit_id")

    common_units = a.index.intersection(b.index)
    paired: list[tuple[float, float]] = []
    for unit_id in common_units:
        sa = a.loc[unit_id, "analysis_score"]
        sb = b.loc[unit_id, "analysis_score"]
        if isinstance(sa, pd.Series):
            sa = sa.iloc[0]
        if isinstance(sb, pd.Series):
            sb = sb.iloc[0]
        if pd.isna(sa) or pd.isna(sb):
            continue
        paired.append((float(sa), float(sb)))

    if len(paired) < 3:
        return None
    x = np.array([p[0] for p in paired])
    y = np.array([p[1] for p in paired])
    return x, y


def _unpaired_samples(
    analysis_df: pd.DataFrame,
    cond_a: str,
    cond_b: str,
    test: str,
    analysis_metric: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    group = analysis_df[
        (analysis_df["test"] == test)
        & (analysis_df["analysis_metric"] == analysis_metric)
    ]
    a = (
        group[group["condition"] == cond_a]["analysis_score"]
        .dropna()
        .to_numpy(dtype=float)
    )
    b = (
        group[group["condition"] == cond_b]["analysis_score"]
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(a) < 3 or len(b) < 3:
        return None
    return a, b


def paired_comparison(
    analysis_df: pd.DataFrame,
    cond_a: str,
    cond_b: str,
    *,
    alpha: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for (test, analysis_metric), group in analysis_df.groupby(
        ["test", "analysis_metric"], dropna=False
    ):
        samples = _paired_samples(analysis_df, cond_a, cond_b, test, analysis_metric)
        if samples is None:
            continue
        x, y = samples

        t_stat, p_value = stats.ttest_rel(x, y)
        diff = x - y
        cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan

        row: dict[str, Any] = {
            "test": test,
            "trait": group["trait"].iloc[0],
            "analysis_metric": analysis_metric,
            "score_scale": group["score_scale"].iloc[0],
            "condition_a": cond_a,
            "condition_b": cond_b,
            "n_pairs": len(x),
            "mean_a": float(x.mean()),
            "mean_b": float(y.mean()),
            "mean_diff_a_minus_b": float(diff.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < alpha),
            "cohens_d_paired": float(cohens_d) if not np.isnan(cohens_d) else np.nan,
            "test_type": "paired_ttest",
        }

        if is_likert_metric(analysis_metric):
            try:
                w_stat, w_p = stats.wilcoxon(x, y, alternative="two-sided")
                row["wilcoxon_statistic"] = float(w_stat)
                row["wilcoxon_p_value"] = float(w_p)
                row["wilcoxon_significant"] = bool(w_p < alpha)
            except ValueError:
                row["wilcoxon_statistic"] = np.nan
                row["wilcoxon_p_value"] = np.nan
                row["wilcoxon_significant"] = False

        results.append(row)
    return results


def unpaired_comparison(
    analysis_df: pd.DataFrame,
    cond_a: str,
    cond_b: str,
    *,
    alpha: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for (test, analysis_metric), group in analysis_df.groupby(
        ["test", "analysis_metric"], dropna=False
    ):
        samples = _unpaired_samples(analysis_df, cond_a, cond_b, test, analysis_metric)
        if samples is None:
            continue
        a, b = samples

        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
        cohens_d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else np.nan

        row: dict[str, Any] = {
            "test": test,
            "trait": group["trait"].iloc[0],
            "analysis_metric": analysis_metric,
            "score_scale": group["score_scale"].iloc[0],
            "condition_a": cond_a,
            "condition_b": cond_b,
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "mean_diff_a_minus_b": float(a.mean() - b.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < alpha),
            "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else np.nan,
            "test_type": "welch_ttest",
        }

        if is_likert_metric(analysis_metric):
            u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
            row["mannwhitney_u"] = float(u_stat)
            row["mannwhitney_p_value"] = float(u_p)
            row["mannwhitney_significant"] = bool(u_p < alpha)

        results.append(row)
    return results


def plot_mean_by_condition(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = summary.dropna(subset=["mean"]).copy()
    if plot_df.empty:
        return

    plot_df["label"] = plot_df["analysis_metric"]
    g = sns.catplot(
        data=plot_df,
        kind="bar",
        x="condition",
        y="mean",
        hue="condition",
        col="label",
        col_wrap=2,
        height=4,
        aspect=1.1,
        legend=False,
        palette="Set2",
        errorbar=None,
    )
    g.set_axis_labels("Condition", "Mean score")
    g.set_titles("{col_name}")
    g.fig.suptitle("Mean scores by condition", y=1.02)
    g.savefig(out_dir / "mean_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)


def plot_condition_comparison(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = summary.dropna(subset=["mean"]).copy()
    if plot_df.empty:
        return

    plot_df["metric"] = plot_df["analysis_metric"]
    pivot = plot_df.pivot_table(
        index="metric",
        columns="condition",
        values="mean",
        aggfunc="first",
    )
    if pivot.empty:
        return

    ax = pivot.plot(kind="bar", figsize=(10, 5), colormap="Set2")
    ax.set_title("Mean primary score by test/trait and condition")
    ax.set_ylabel("Mean score")
    ax.set_xlabel("Metric")
    ax.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(out_dir / "grouped_bar_comparison.png", dpi=150)
    plt.close()


def _plot_pvalue_heatmap(
    tests_df: pd.DataFrame,
    *,
    test_type: str,
    p_col: str,
    title: str,
    cbar_label: str,
    out_path: Path,
) -> None:
    if tests_df.empty or "test_type" not in tests_df.columns:
        return
    if p_col not in tests_df.columns:
        return

    subset = tests_df[tests_df["test_type"] == test_type].copy()
    subset = subset.dropna(subset=[p_col])
    if subset.empty:
        return

    subset["metric"] = subset["analysis_metric"]
    subset["comparison"] = subset["condition_a"] + " vs " + subset["condition_b"]
    pivot = subset.pivot_table(
        index="metric",
        columns="comparison",
        values=p_col,
        aggfunc="first",
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.5)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.1,
        ax=ax,
        cbar_kws={"label": cbar_label},
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pvalue_heatmaps(tests_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_pvalue_heatmap(
        tests_df,
        test_type="paired_ttest",
        p_col="p_value",
        title="Paired t-test p-values (lower = more different)",
        cbar_label="p-value (paired t-test)",
        out_path=out_dir / "pvalue_heatmap.png",
    )
    _plot_pvalue_heatmap(
        tests_df,
        test_type="paired_ttest",
        p_col="wilcoxon_p_value",
        title="Wilcoxon signed-rank p-values — Likert metrics (paired)",
        cbar_label="p-value (Wilcoxon signed-rank)",
        out_path=out_dir / "pvalue_heatmap_wilcoxon.png",
    )
    _plot_pvalue_heatmap(
        tests_df,
        test_type="welch_ttest",
        p_col="mannwhitney_p_value",
        title="Mann–Whitney U p-values — Likert metrics (unpaired)",
        cbar_label="p-value (Mann–Whitney U)",
        out_path=out_dir / "pvalue_heatmap_mannwhitney.png",
    )


def plot_transparency_breakdown(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tdf = df[df["trait"] == "transparency"].copy()
    if tdf.empty:
        return

    pairs = (
        tdf.dropna(subset=["pair_id", "pair_invariance_score"])
        .drop_duplicates(subset=["condition", "pair_id"])
    )
    if not pairs.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=pairs, x="condition", y="pair_invariance_score", ax=ax)
        sns.stripplot(
            data=pairs,
            x="condition",
            y="pair_invariance_score",
            color="black",
            alpha=0.4,
            ax=ax,
        )
        ax.set_title("Transparency: framing invariance by condition")
        ax.set_ylabel("Pair invariance score (1–5)")
        plt.tight_layout()
        plt.savefig(out_dir / "transparency_invariance.png", dpi=150)
        plt.close()

    gap = tdf.dropna(subset=["candor_framing_gap"]).drop_duplicates(
        subset=["condition", "pair_id"]
    )
    if not gap.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=gap, x="condition", y="candor_framing_gap", ax=ax)
        ax.set_title("Transparency: candor score gap (eval vs deploy)")
        ax.set_ylabel("|eval candor − deploy candor|")
        plt.tight_layout()
        plt.savefig(out_dir / "transparency_candor_gap.png", dpi=150)
        plt.close()


def plot_capability_judge(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = summary[summary["analysis_metric"] == "capability"].copy()
    if cap.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=cap,
        x="condition",
        y="mean",
        hue="condition",
        ax=ax,
        palette="Set2",
        legend=False,
    )
    for idx, (_, row) in enumerate(cap.iterrows()):
        ax.text(
            idx,
            row["mean"] + 0.05,
            f"n={int(row['n'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, 5.5)
    ax.set_title("Test 0 capability: mean Prometheus judge score by condition")
    ax.set_ylabel("Mean judge score (1–5)")
    plt.tight_layout()
    plt.savefig(out_dir / "test0_judge_score.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze judged eval outputs.")
    parser.add_argument("--conditions", default="A,B,C")
    parser.add_argument("--judged-dir", type=Path, default=JUDGED_DIR)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help=(
            "Base directory for analysis outputs (default: evals/reports). "
            "If default outputs already exist and this flag is left at the "
            "default, a fresh run_NNN subdirectory is created automatically."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    explicit_reports = args.reports_dir.resolve() != REPORTS_DIR.resolve()
    reports_dir = allocate_reports_dir(args.reports_dir, explicit=explicit_reports)
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if reports_dir != args.reports_dir.resolve():
        print(f"Existing reports found under {args.reports_dir}; writing new run to {reports_dir}")

    print(f"Loading judged outputs for conditions: {conditions}")
    raw_df = load_judged(conditions, args.judged_dir)
    analysis_df = to_analysis_long(raw_df)

    scored = analysis_df["analysis_score"].notna().sum()
    print(f"Loaded {len(raw_df)} judged rows -> {len(analysis_df)} analysis units")
    print(f"  ({scored} with scores)")

    summary = summarize(analysis_df)
    summary_path = reports_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    paired_rows: list[dict[str, Any]] = []
    welch_rows: list[dict[str, Any]] = []
    for cond_a, cond_b in combinations(conditions, 2):
        paired_rows.extend(
            paired_comparison(analysis_df, cond_a, cond_b, alpha=args.alpha)
        )
        welch_rows.extend(
            unpaired_comparison(analysis_df, cond_a, cond_b, alpha=args.alpha)
        )

    paired_df = pd.DataFrame(paired_rows)
    welch_df = pd.DataFrame(welch_rows)

    tests_path = reports_dir / "pairwise_tests.csv"
    all_tests = pd.concat([paired_df, welch_df], ignore_index=True)
    all_tests.to_csv(tests_path, index=False)
    print(f"Wrote {tests_path}")

    sns.set_theme(style="whitegrid")
    plot_mean_by_condition(summary, figures_dir)
    plot_condition_comparison(summary, figures_dir)
    plot_pvalue_heatmaps(all_tests, figures_dir)
    plot_capability_judge(summary, figures_dir)
    plot_transparency_breakdown(raw_df, figures_dir)
    print(f"Wrote figures to {figures_dir}")

    bundle = {
        "conditions": conditions,
        "alpha": args.alpha,
        "summary": summary.to_dict(orient="records"),
        "paired_tests": paired_df.to_dict(orient="records"),
        "welch_tests": welch_df.to_dict(orient="records"),
    }
    stats_path = reports_dir / "stats.json"
    stats_path.write_text(json.dumps(bundle, indent=2))
    print(f"Wrote {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
