"""
Plot the per-class precision-recall curves from scripts/compute_pr_curves.py
as PNG images, one chart per class, production vs new_model overlaid, with
each source's best-F1 threshold marked on its curve.

Reads:
    reports/pr_curves.json   (output of scripts/compute_pr_curves.py)

Writes:
    <output>/pr_curve_<class>.png   - one plot per class

Usage:
    python scripts/plot_pr_curves.py --report reports/pr_curves.json --output reports/pr_curve_plots
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import setup_logger, get_logger

SOURCE_COLORS = {
    "production": "#3498db",
    "new_model": "#e67e22",
}


def plot_class(cls: str, sources: dict, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))

    for source, color in SOURCE_COLORS.items():
        data = sources.get(source, {}).get(cls)
        if not data or not data["curve"]:
            continue

        recalls = [p["recall"] for p in data["curve"]]
        precisions = [p["precision"] for p in data["curve"]]
        ax.plot(recalls, precisions, color=color, linewidth=2, label=source, marker=".", markersize=3)

        best = data["best_f1"]
        ax.scatter(
            [best["recall"]], [best["precision"]],
            color=color, edgecolor="black", zorder=5, s=90, marker="*",
        )
        ax.annotate(
            f"  best F1={best['f1']:.2f}\n  @thr={best['threshold']:.2f}",
            (best["recall"], best["precision"]),
            fontsize=8, color=color, va="center",
        )

    num_gt = next(
        (sources[s][cls]["num_gt"] for s in SOURCE_COLORS if cls in sources.get(s, {})),
        None,
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall: {cls}" + (f"  ({num_gt} GT boxes)" if num_gt is not None else ""))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left")
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_all(report_path: str, output_dir: str) -> list:
    logger = get_logger(__name__)

    with open(report_path, "r") as f:
        report = json.load(f)

    sources = report["sources"]
    classes = sorted({cls for src in sources.values() for cls in src.keys()})

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written = []
    for cls in classes:
        png_path = output_path / f"pr_curve_{cls}.png"
        plot_class(cls, sources, png_path)
        written.append(png_path)
        logger.info(f"Wrote {png_path}")

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", default="reports/pr_curves.json", help="Input PR-curve JSON (from compute_pr_curves.py)")
    parser.add_argument("--output", default="reports/pr_curve_plots", help="Output directory for PNG plots")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(log_dir=args.log_dir)

    written = plot_all(args.report, args.output)

    print(f"Wrote {len(written)} plot(s) to {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
