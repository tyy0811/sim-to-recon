"""Generate V1 figures: scatter plot of all runs per view count, medians overlaid."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Consistent seed colors so the reader can track a seed across panels
SEED_COLORS = {
    42: "#1f77b4",   # blue
    123: "#ff7f0e",  # orange
    7: "#2ca02c",    # green
}


def plot_variance_scatter(runs: list[dict], output_path: Path) -> None:
    """Scatter plot: X=view count, Y=metric, one dot per (N, seed) + median overlay.

    Three panels: points, chamfer, F@5mm. Each panel shows the full
    within-N variance rather than error bars on a line, to correctly
    represent bimodal success/failure distributions at low N.
    """
    by_n = defaultdict(list)
    for r in runs:
        n = r.get("n_views")
        if n is None:
            continue
        by_n[n].append(r)

    view_counts = sorted(by_n.keys(), reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panels = [
        ("n_points", "Fused Points", "log", axes[0]),
        ("chamfer", "Chamfer Distance (mm)", "linear", axes[1]),
        ("f_score_5mm", "F-score @ 5mm", "linear", axes[2]),
    ]

    for key, label, yscale, ax in panels:
        # Plot per-seed scatter points
        for n in view_counts:
            for r in by_n[n]:
                val = r.get(key)
                if val is None or val == 0:
                    # Show degenerate runs at the plot floor (log scale can't handle 0)
                    if yscale == "log":
                        val = 1
                    else:
                        val = 0
                seed = r.get("seed")
                color = SEED_COLORS.get(seed, "#808080")
                ax.scatter(n, val, s=140, color=color, alpha=0.75,
                           edgecolors="black", linewidths=0.8, zorder=3)

        # Overlay medians as a black line
        medians_x, medians_y = [], []
        for n in view_counts:
            vals = [r.get(key) for r in by_n[n] if r.get(key) is not None and r.get(key) > 0]
            if vals:
                medians_x.append(n)
                medians_y.append(float(np.median(vals)))
        ax.plot(medians_x, medians_y, "k-", linewidth=2, alpha=0.6,
                label="median", zorder=2)
        ax.scatter(medians_x, medians_y, s=60, color="black", marker="_",
                   linewidths=2.5, zorder=4)

        ax.set_xlabel("Number of Views", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3, zorder=1)
        ax.invert_xaxis()
        ax.set_xticks(view_counts)

    # Legend (only on the first panel) with seed colors
    handles = [
        plt.scatter([], [], s=140, color=c, edgecolors="black", linewidths=0.8,
                    label=f"seed {s}")
        for s, c in SEED_COLORS.items()
    ]
    handles.append(plt.Line2D([], [], color="black", linewidth=2, alpha=0.6,
                              label="median"))
    axes[0].legend(handles=handles, loc="lower left", fontsize=10, framealpha=0.9)

    plt.suptitle(
        "Reconstruction quality vs view count — 3 seeds per N  "
        "(within-N variance, not a clean degradation curve)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved variance scatter: {output_path}")


def plot_failure_gallery(
    results_dir: Path, runs: list[dict], output_path: Path
) -> None:
    """Plot per-point error heatmaps for the best (highest F@5mm) run per N.

    Uses the saved per_point_errors.npz and dense.ply from each sweep run.
    Renders 3 orthographic projections per view count.
    """
    import open3d as o3d

    from simtorecon.evaluation.alignment import align_to_gt

    # Find the best run (highest F@5mm) per view count
    by_n = defaultdict(list)
    for r in runs:
        if r.get("n_views") is None or r.get("f_score_5mm") is None:
            continue
        by_n[r["n_views"]].append(r)

    best_per_n = []
    for n in sorted(by_n.keys(), reverse=True):
        best = max(by_n[n], key=lambda r: r["f_score_5mm"] or 0)
        seed = best.get("seed")
        run_dir = results_dir / f"views_{n:03d}" / f"seed_{seed}"
        npz_path = run_dir / "per_point_errors.npz"
        ply_path = run_dir / "dense.ply"
        if npz_path.exists() and ply_path.exists():
            best_per_n.append((n, seed, run_dir, best))

    if not best_per_n:
        print("No per-point error files found — skipping failure gallery.")
        return

    n_runs = len(best_per_n)
    fig, axes = plt.subplots(n_runs, 3, figsize=(15, 4 * n_runs))
    if n_runs == 1:
        axes = axes[np.newaxis, :]

    gt_ply = results_dir / "gt.ply"
    gt = o3d.io.read_point_cloud(str(gt_ply)) if gt_ply.exists() else None

    for row, (n_views, seed, run_dir, result) in enumerate(best_per_n):
        errors = np.load(run_dir / "per_point_errors.npz")["errors"]
        pred = o3d.io.read_point_cloud(str(run_dir / "dense.ply"))

        if gt is not None:
            aligned, _, _ = align_to_gt(pred, gt, threshold=20.0, max_iterations=200)
            pts = np.asarray(aligned.points)
        else:
            pts = np.asarray(pred.points)

        if len(pts) == 0:
            for col in range(3):
                axes[row, col].set_title(f"n={n_views} (seed {seed}) — DEGENERATE",
                                         fontsize=11)
            continue

        e_clip = np.clip(errors, 0, np.percentile(errors, 95))

        projections = [
            (0, 1, "XY (top)"),
            (0, 2, "XZ (front)"),
            (1, 2, "YZ (side)"),
        ]
        for col, (xi, yi, proj_label) in enumerate(projections):
            ax = axes[row, col]
            sc = ax.scatter(
                pts[:, xi], pts[:, yi], c=e_clip, cmap="RdYlBu_r",
                s=0.3, alpha=0.6, rasterized=True,
            )
            ax.set_aspect("equal")
            title = f"n={n_views} (seed {seed}, best of 3) — {proj_label}"
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=8)
            if col == 2:
                cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Error (mm)", fontsize=9)

    plt.suptitle("Per-Point Error — Best Seed per View Count",
                 fontsize=14, y=1.00)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved failure gallery: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate V1 figures")
    parser.add_argument("--runs", default="results/stress_view_count/all_runs.json")
    parser.add_argument("--output-dir", default="docs/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_path = Path(args.runs)
    if not runs_path.exists():
        print(f"No runs found at {runs_path}. Run stress sweep first.")
        return

    with open(runs_path) as f:
        runs = json.load(f)

    print(f"Loaded {len(runs)} runs")

    # Figure 1: variance scatter (the primary figure)
    plot_variance_scatter(runs, output_dir / "variance_scatter.png")

    # Figure 2: failure region gallery on best-of-3 per N
    results_dir = runs_path.parent
    plot_failure_gallery(results_dir, runs, output_dir / "failure_gallery.png")


if __name__ == "__main__":
    main()
