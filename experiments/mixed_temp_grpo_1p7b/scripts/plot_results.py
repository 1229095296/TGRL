from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.mixed_temp_grpo_1p7b.scripts.analysis_helpers import format_report_section, load_table


def load_optional_table(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    return load_table(path)


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=200)
    fig.savefig(output_dir / f"{stem}.pdf")


def plot_metric_curves(df: pd.DataFrame, x_col: str, y_col: str, group_col: str, output_dir: Path, stem: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for group_value, group_df in df.groupby(group_col):
        group_df = group_df.sort_values(x_col)
        ax.plot(group_df[x_col], group_df[y_col], marker="o", label=str(group_value))
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    _save_figure(fig, output_dir, stem)
    plt.close(fig)


def plot_summary_bars(
    df: pd.DataFrame,
    value_cols: list[str],
    label_col: str,
    output_dir: Path,
    stem: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    value_cols = [col for col in value_cols if col in df.columns]
    if not value_cols:
        return

    plot_df = df.copy()
    plot_df[label_col] = plot_df[label_col].astype(str)
    fig, axes = plt.subplots(1, len(value_cols), figsize=(6 * len(value_cols), 5))
    axes = [axes] if len(value_cols) == 1 else list(axes)
    for ax, value_col in zip(axes, value_cols):
        ax.bar(plot_df[label_col], plot_df[value_col])
        ax.set_title(value_col)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(title)
    _save_figure(fig, output_dir, stem)
    plt.close(fig)


def build_report(
    final_summary: pd.DataFrame,
    gain_identification: pd.DataFrame,
    carrier_expectation: pd.DataFrame,
    output_dir: Path,
) -> None:
    sections: list[str] = ["# Mixed-Temperature GRPO Offline Analysis Report", ""]

    if not final_summary.empty:
        sort_col = "pass_at_1" if "pass_at_1" in final_summary.columns else final_summary.columns[-1]
        best_row = final_summary.sort_values(sort_col, ascending=False).iloc[0]
        bullets = [
            f"final_summary 行数: {len(final_summary)}",
            f"最佳 {sort_col}: {best_row.get(sort_col, 'n/a')}",
            f"最佳配置: method={best_row.get('method', 'unknown')}, checkpoint={best_row.get('checkpoint', 'unknown')}, seed={best_row.get('seed', 'unknown')}",
        ]
        for col in ["pass_at_8", "pass_at_16", "pearson_hat_vs_true", "auroc_true_delta_positive", "carrier_gap"]:
            if col in best_row.index:
                bullets.append(f"{col}: {best_row[col]}")
        sections.append(format_report_section("Final Summary", bullets))

    if not gain_identification.empty:
        gain_row = gain_identification.sort_values("pearson_hat_vs_true", ascending=False).iloc[0]
        bullets = [
            f"gain_identification 行数: {len(gain_identification)}",
            f"最佳 Pearson: {gain_row.get('pearson_hat_vs_true', 'n/a')}",
            f"对应 method={gain_row.get('method', 'unknown')}, checkpoint={gain_row.get('checkpoint', 'unknown')}, seed={gain_row.get('seed', 'unknown')}",
        ]
        for col in ["spearman_hat_vs_true", "auroc_true_delta_positive", "sign_accuracy", "positive_hit_rate", "negative_hit_rate"]:
            if col in gain_row.index:
                bullets.append(f"{col}: {gain_row[col]}")
        sections.append(format_report_section("Gain Identification", bullets))

    if not carrier_expectation.empty:
        carrier_row = carrier_expectation.sort_values("carrier_gap", ascending=False).iloc[0]
        bullets = [
            f"carrier_expectation 行数: {len(carrier_expectation)}",
            f"最大 carrier_gap: {carrier_row.get('carrier_gap', 'n/a')}",
            f"对应 method={carrier_row.get('method', 'unknown')}, checkpoint={carrier_row.get('checkpoint', 'unknown')}, seed={carrier_row.get('seed', 'unknown')}",
        ]
        for col in ["carrier_mean_true_positive", "carrier_mean_true_negative"]:
            if col in carrier_row.index:
                bullets.append(f"{col}: {carrier_row[col]}")
        sections.append(format_report_section("Carrier Expectation", bullets))

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and summarize mixed-temp GRPO offline outputs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics", default=None, help="metrics.jsonl from training.")
    parser.add_argument("--eval-summary", default=None, help="checkpoint_eval_summary.csv")
    parser.add_argument("--gain-identification", default=None, help="gain_identification.csv")
    parser.add_argument("--carrier-expectation", default=None, help="carrier_expectation.csv")
    parser.add_argument("--final-summary", default=None, help="final_summary.csv")
    parser.add_argument("--curve-group-col", default="method")
    parser.add_argument("--curve-x-col", default="training/global_step")
    parser.add_argument("--curve-y-col", default=None, help="Override training curve y-axis column.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_optional_table(args.metrics)
    eval_summary = load_optional_table(args.eval_summary)
    gain_identification = load_optional_table(args.gain_identification)
    carrier_expectation = load_optional_table(args.carrier_expectation)
    final_summary = load_optional_table(args.final_summary)

    if not metrics.empty and args.curve_x_col in metrics.columns:
        y_candidates = [args.curve_y_col] if args.curve_y_col else [
            "reward/gap",
            "reward/explore_mean",
            "reward/control_mean",
            "critic/rewards/mean",
        ]
        y_col = next((col for col in y_candidates if col and col in metrics.columns), None)
        if y_col is not None and args.curve_group_col in metrics.columns:
            plot_metric_curves(
                metrics,
                x_col=args.curve_x_col,
                y_col=y_col,
                group_col=args.curve_group_col,
                output_dir=output_dir,
                stem="training_curves",
            )

    if not gain_identification.empty and "method" in gain_identification.columns:
        plot_summary_bars(
            gain_identification,
            value_cols=[
                "pearson_hat_vs_true",
                "auroc_true_delta_positive",
                "sign_accuracy",
            ],
            label_col="method",
            output_dir=output_dir,
            stem="gain_identification",
            title="Gain Identification",
        )

    if not carrier_expectation.empty and "method" in carrier_expectation.columns:
        plot_summary_bars(
            carrier_expectation,
            value_cols=[
                "carrier_mean_true_positive",
                "carrier_mean_true_negative",
                "carrier_gap",
            ],
            label_col="method",
            output_dir=output_dir,
            stem="carrier_expectation",
            title="Carrier Expectation",
        )

    if not final_summary.empty and "method" in final_summary.columns:
        plot_summary_bars(
            final_summary,
            value_cols=["pass_at_1", "pass_at_8", "pass_at_16"],
            label_col="method",
            output_dir=output_dir,
            stem="final_summary",
            title="Final Summary",
        )

    build_report(
        final_summary=final_summary if not final_summary.empty else eval_summary,
        gain_identification=gain_identification,
        carrier_expectation=carrier_expectation,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
