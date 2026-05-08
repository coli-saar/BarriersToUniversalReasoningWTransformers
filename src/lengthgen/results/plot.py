import fire
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from lengthgen.paths import RESULTS_OUT_BASE_PATH
from pathlib import Path

RESULTS_PATH = RESULTS_OUT_BASE_PATH / "test_results"
FIGURES_PATH = RESULTS_OUT_BASE_PATH / "figures"

GREEN = "#2E8B57"
RED = "#C44E52"

CONDITION_STYLE = {
    "index": {"color": GREEN, "label": "Signpots",    "linestyle": "-", "linewidth": 2.8, "alpha": 0.10},
    "delta": {"color": GREEN, "label": "Value change", "linestyle": "-", "linewidth": 2.8, "alpha": 0.10},
    "value_change": {"color": GREEN, "label": "Value change", "linestyle": "-", "linewidth": 2.8, "alpha": 0.10},
    "base": {"color": RED, "label": "Naive CoT", "linestyle": "--", "linewidth": 2.4, "alpha": 0.08},
}

def setup_style():
    sns.set_theme(style="whitegrid", context="paper")

    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 10.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.16,
        "lines.linewidth": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def style_axes(ax: plt.Axes) -> None:
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.tick_params(axis="both", width=0.9, length=3.5)

def load_condition_results(task, train_len, pos_encodings, cond_dict):
    cond_name = cond_dict["name"]
    repetitive = cond_dict.get("repetitive", False)
    if repetitive:
        base = RESULTS_PATH / pos_encodings / "repetitive" / task / f"len{train_len}" / cond_name
    else:
        base = RESULTS_PATH / pos_encodings / task / f"len{train_len}" / cond_name
    if not base.exists():
        return {}

    per_length = {}
    for csv_file in sorted(base.glob("*/full_results_*.csv")):
        df = pd.read_csv(csv_file)
        cut_off = int(train_len*2.0)
        df = df[df["Length"] <= cut_off]
        for _, row in df.iterrows():
            length = int(row["Length"])
            acc    = float(row["Accuracy"])
            per_length.setdefault(length, []).append(acc)

    return per_length

def compute_stats(per_length):
    lengths = sorted(per_length.keys())
    means = [np.mean(per_length[l]) for l in lengths]
    mins = [np.min(per_length[l]) for l in lengths]
    maxs = [np.max(per_length[l]) for l in lengths]
    return lengths, means, mins, maxs

def plot_task_train_len(task, train_len, pos_encodings, conditions, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))

    for cond in conditions:
        cond_name = cond["name"]
        per_length = load_condition_results(task, train_len, pos_encodings, cond)
        if not per_length:
            print(f"No data for {task} len{train_len} {cond_name} — skipping")
            continue

        style = CONDITION_STYLE.get(cond_name, {
            "color": "#9E9E9E", 
            "label": cond_name.replace("_", " ").capitalize(), 
            "linestyle": "-.",
            "linewidth": 2.5,
            "alpha": 0.10
        })
        lengths, means, mins, maxs = compute_stats(per_length)

        ax.plot(
            lengths, means,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
            linewidth=style.get("linewidth", 2.5),
            zorder=3,
        )
        
        ax.fill_between(
            lengths, mins, maxs,
            color=style["color"], alpha=style.get("alpha", 0.15),
            linewidth=0,
            zorder=2,
        )

    # Mark training length boundary
    ax.axvline(
        x=train_len, color="#666666", linestyle=":",
        linewidth=1.0, alpha=0.85, zorder=1, label=f"Train length ({train_len})"
    )
    ax.set_xlabel("Sequence length", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title(f"{task.capitalize()}  —  trained up to length {train_len}", pad=6, fontweight="semibold")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9.5, frameon=True, fancybox=True, framealpha=0.96, edgecolor="#D9D9D9")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    style_axes(ax)

    fig.tight_layout()
    fname = out_dir / f"{task}_len{train_len}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_all_train_lens(task, train_lengths, pos_encodings, conditions, out_dir):
    n = len(train_lengths)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, train_len in zip(axes, train_lengths):
        for cond in conditions:
            cond_name = cond["name"]
            per_length = load_condition_results(task, train_len, pos_encodings, cond)
            if not per_length:
                continue

            style = CONDITION_STYLE.get(cond_name, {
                "color": "#9E9E9E", 
                "label": cond_name.replace("_", " ").capitalize(), 
                "linestyle": "-.",
                "linewidth": 2.5,
                "alpha": 0.10
            })
            lengths, means, mins, maxs = compute_stats(per_length)

            ax.plot(
                lengths, means,
                color=style["color"],
                linestyle=style["linestyle"],
                label=style["label"],
                linewidth=style.get("linewidth", 2.5),
                zorder=3,
            )
            ax.fill_between(
                lengths, mins, maxs,
                color=style["color"], alpha=style.get("alpha", 0.15),
                linewidth=0,
                zorder=2,
            )

        ax.axvline(x=train_len, color="#666666", linestyle=":", linewidth=1.0, alpha=0.85, zorder=1)
        ax.set_title(f"Train len {train_len}", pad=6, fontweight="semibold")
        ax.set_xlabel("Sequence length", fontsize=9.5, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")

        style_axes(ax)

    axes[0].set_ylabel("Accuracy", fontweight="bold")

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9.5, frameon=True, fancybox=True, framealpha=0.96, edgecolor="#D9D9D9")
    fig.suptitle(task.capitalize(), fontsize=14, y=1.0, fontweight="bold")
    fig.tight_layout()
    fname = out_dir / f"{task}_all_lens.png"
    fig.savefig(fname)
    plt.close(fig)
    print(f"Saved {fname}")

def main(
    config: str,
    task: str = None,
):
    setup_style()
    with open(config) as f:
        cfg = yaml.safe_load(f)

    conditions = [c for c in cfg["conditions"]]
    tasks_to_run = [task] if task else list(cfg["tasks"].keys())
 
    for t in tasks_to_run:
        print(f"\nPlotting {t}...")
        task_cfg = cfg["tasks"][t]
        train_lengths = task_cfg["train_lengths"]
        repetitive = task_cfg.get("repetitive",False)
        fig_pos_path = FIGURES_PATH / task_cfg['positional_encodings']
        fig_pos_path.mkdir(parents=True, exist_ok=True)
        # One plot per train_len
        for train_len in train_lengths:
            plot_task_train_len(t, train_len, task_cfg['positional_encodings'], conditions, fig_pos_path)
 
        # Combined plot across all train_lens
        if len(train_lengths) > 1:
            plot_all_train_lens(t, train_lengths, task_cfg['positional_encodings'], conditions, fig_pos_path)
 
if __name__ == "__main__":
    fire.Fire(main)