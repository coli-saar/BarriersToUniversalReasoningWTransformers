#!/usr/bin/env python3
"""
Multi-scheme error analysis for algorithmic length generalization.
"""

import os
import json
import re
import csv
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from lengthgen.constants import TRACE_TOKEN, FINAL_ANSWER_TOKEN, END_OF_TEXT_TOKEN

# ─── Scheme configuration ─────────────────────────────────────────────────────

SCHEME_PARAMS = {
    "naive":     {"delta_cot": False, "include_indices": False},
    "signposts": {"delta_cot": False, "include_indices": True},
    "delta_cot": {"delta_cot": True,  "include_indices": True},
}

# ─── Error taxonomy ───────────────────────────────────────────────────────────

ERROR_CATEGORIES = [
    "Trace: Wrong Index Loaded",
    "Trace: Wrong Swap Boxes",
    "Trace: Incorrect State Write",
    "Trace: Wrong W/K Token",
    "Trace: Incorrect Transition Value",
    "Trace: Truncated/Missing Steps",
    "Trace: Hallucinated Extra Steps",
    "Res: Wrong Relation or Final Value",
    "Answer: Wrong Despite Correct Trace",
    "Unknown Error",
]

# ─── Style ────────────────────────────────────────────────────────────────────

def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi":        200,
        "savefig.dpi":       300,
        "font.size":         10.5,
        "axes.titlesize":    10.5,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.9,
        "grid.linewidth":    0.5,
        "grid.alpha":        0.16,
        "lines.linewidth":   2.5,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def style_axes(ax: plt.Axes) -> None:
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.tick_params(axis="both", width=0.9, length=3.5)


_COLORS  = plt.cm.tab10.colors
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "+"]

CAT_STYLE = {
    cat: {"color": _COLORS[i % len(_COLORS)], "marker": _MARKERS[i % len(_MARKERS)]}
    for i, cat in enumerate(ERROR_CATEGORIES)
}

# ─── Ground truth reconstruction ──────────────────────────────────────────────

def generate_gt_from_prompt(prompt: str, delta_cot: bool = True, include_indices: bool = True) -> str:
    init_part, ops_part = prompt.split("operation")

    init_tokens = re.findall(r"([A-Z])\s+(Cat|Dog)", init_part)
    state = {k: v for k, v in init_tokens}
    init_state = state.copy()

    ops = re.findall(r"(<\d+>)?\s*swap\s+([A-Z])\s+([A-Z])", ops_part)

    trace_steps = []
    history = {k: [] for k in state.keys()}

    for curr_id, box1, box2 in ops:
        curr_id = curr_id.strip() if curr_id else ""
        val_i = state[box1]
        val_j = state[box2]
        label = f"{curr_id} " if include_indices and curr_id else ""

        if delta_cot:
            action_tokens = []
            if val_i != val_j:
                action_tokens.append(f"W_{box1} {val_i}_{val_j}")
                action_tokens.append(f"W_{box2} {val_j}_{val_i}")
                history[box1].append((val_i, val_j))
                history[box2].append((val_j, val_i))
            else:
                action_tokens.append(f"K_{box1}")
                action_tokens.append(f"K_{box2}")

            action_str = " ".join(action_tokens)
            step_str = f"load {curr_id} . {label}swap {box1} {box2} {action_str} "
        else:
            temp_state = state.copy()
            temp_state[box1], temp_state[box2] = temp_state[box2], temp_state[box1]
            indexed_state = " ".join(f"{k} {temp_state[k]}" for k in sorted(temp_state.keys()))

            if include_indices:
                step_str = f"load {curr_id} . {label}swap {box1} {box2} write {indexed_state} "
            else:
                step_str = f"swap {box1} {box2} write {indexed_state} . "

        trace_steps.append(step_str)
        state[box1], state[box2] = state[box2], state[box1]

    trace_str = "".join(trace_steps) + "end "

    if delta_cot:
        res_parts = ["res"]
        for var_name in sorted(state.keys()):
            initial_val  = init_state[var_name]
            transitions  = history[var_name]
            var_res      = f"<{var_name}> init {initial_val}"
            if not transitions:
                var_res += f" IN == OUT final {initial_val}"
            else:
                in_count  = sum(1 for (old, new) in transitions if new == initial_val)
                out_count = sum(1 for (old, new) in transitions if old == initial_val)
                relation  = "==" if in_count == out_count else "<"
                final_val = initial_val if relation == "==" else transitions[-1][1]
                var_res  += f" IN {relation} OUT final {final_val}"
            res_parts.append(var_res)
        trace_str += " ".join(res_parts) + " "

    answer_str = " ".join(state[k] for k in sorted(state.keys()))
    return f"{prompt} {trace_str}{FINAL_ANSWER_TOKEN} {answer_str}"


# ─── Parsing ──────────────────────────────────────────────────────────────────

def parse_trace(text: str, scheme: str) -> dict:
    completion = text.split(TRACE_TOKEN, 1)[-1].replace(END_OF_TEXT_TOKEN, "").strip()

    answer = ""
    if f"{FINAL_ANSWER_TOKEN} " in completion:
        completion, answer = completion.rsplit(f"{FINAL_ANSWER_TOKEN} ", 1)
        answer = answer.strip()

    res = ""
    if " end res " in completion:
        completion, res = completion.split(" end res ", 1)
        res = "res " + res.strip()
    elif " end " in completion:
        completion = re.split(r"\bend\b", completion)[0]

    if scheme in ("signposts", "delta_cot"):
        parts = completion.split("load ")
        steps = [
            "load " + p.strip()
            for p in parts
            if p.strip() and not re.match(r"^end\b", p.strip())
        ]
    else:
        parts = completion.split(". ")
        steps = [
            p.strip()
            for p in parts
            if p.strip() and not re.match(r"^(load\s+)?end\b", p.strip())
        ]

    return {"steps": steps, "res": res, "answer": answer}


def parse_step_components(step: str, scheme: str) -> dict:
    out = {}
    if scheme in ("signposts", "delta_cot"):
        m = re.search(r"load\s+(<\d+>)", step)
        out["index"] = m.group(1) if m else None

    m = re.search(r"swap\s+([A-Z])\s+([A-Z])", step)
    if m:
        out["box1"] = m.group(1)
        out["box2"] = m.group(2)

    if scheme == "delta_cot":
        wk_tokens = re.findall(r"[WK]_[A-Z](?:\s+\S+)?", step)
        out["wk_types"]  = [t[0] for t in wk_tokens]
        out["wk_tokens"] = [t.strip() for t in wk_tokens]
    else:
        m = re.search(r"write\s+(.+?)(?:\s*\.|$)", step)
        out["written_state"] = m.group(1).strip() if m else None

    return out


# ─── Error categorisation ─────────────────────────────────────────────────────

def categorize_step_error(pred_step: str, gt_step: str, scheme: str):
    if pred_step == gt_step:
        return None

    pred = parse_step_components(pred_step, scheme)
    gt   = parse_step_components(gt_step,   scheme)

    if scheme in ("signposts", "delta_cot"):
        if pred.get("index") != gt.get("index"):
            return "Trace: Wrong Index Loaded"

    if pred.get("box1") != gt.get("box1") or pred.get("box2") != gt.get("box2"):
        return "Trace: Wrong Swap Boxes"

    if scheme == "delta_cot":
        if pred.get("wk_types") != gt.get("wk_types"):
            return "Trace: Wrong W/K Token"
        return "Trace: Incorrect Transition Value"

    return "Trace: Incorrect State Write"


def categorize_error(pred_text: str, gt_text: str, scheme: str) -> str:
    pred = parse_trace(pred_text, scheme)
    gt   = parse_trace(gt_text,   scheme)

    for i, gt_step in enumerate(gt["steps"]):
        if i >= len(pred["steps"]):
            return "Trace: Truncated/Missing Steps"
        cat = categorize_step_error(pred["steps"][i], gt_step, scheme)
        if cat:
            return cat

    if len(pred["steps"]) > len(gt["steps"]):
        return "Trace: Hallucinated Extra Steps"

    if scheme == "delta_cot" and pred["res"] != gt["res"]:
        return "Res: Wrong Relation or Final Value"

    if pred["answer"] != gt["answer"]:
        return "Answer: Wrong Despite Correct Trace"

    return "Unknown Error"


# ─── Folder-level analysis ────────────────────────────────────────────────────

def extract_length(fname: str):
    m = re.search(r"_len(\d+)\_repetitive\.json$", fname)
    return int(m.group(1)) if m else None


def analyze_folder(folder: str, scheme: str) -> dict:
    params  = SCHEME_PARAMS[scheme]
    results = {}

    fnames = sorted(
        [f for f in os.listdir(folder) if f.endswith(".json")],
        key=lambda f: extract_length(f) or 0
    )

    for fname in fnames:
        length = extract_length(fname)
        if length is None:
            continue

        with open(os.path.join(folder, fname)) as f:
            data = json.load(f)

        incorrect = [x for x in data if not x.get("correct", True)]
        counter   = Counter()

        for item in incorrect:
            # Attempt to use exactly what the original generator outputted
            if "gt_full_text" in item:
                gt_full = item["gt_full_text"]
            else:
                gt_full  = generate_gt_from_prompt(item["prompt"], **params)
                
            category = categorize_error(item["full_output"], gt_full, scheme)
            counter[category] += 1

        results[length] = counter

        total = len(data)
        n_err = len(incorrect)
        print(f"  [{scheme:10s}] len={length:3d} | {n_err:4d}/{total} errors | {dict(counter)}")

    return results


# ─── Data Export ──────────────────────────────────────────────────────────────

def export_to_csv(models: list, all_data: list, output_path: str):
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Test Length", "Error Category", "Count"])
        for cfg, data in zip(models, all_data):
            for length in sorted(data.keys()):
                for cat in ERROR_CATEGORIES:
                    count = data[length].get(cat, 0)
                    if count > 0:
                        writer.writerow([cfg["name"], length, cat, count])
    print(f"\nError data successfully saved to: {output_path}")


# ─── Shared legend helper ─────────────────────────────────────────────────────

def _make_legend(fig: plt.Figure, handles: list, n_cols: int = 3) -> None:
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), n_cols),
        fontsize=9,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor="#D9D9D9",
        handlelength=2.8,
        borderpad=0.45,
        labelspacing=0.4,
        bbox_to_anchor=(0.5, 0.02),
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_linewidth(0.8)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_error_evolution(
    models: list,
    all_data: list,
    output_image_path: str = None,
    output_csv_path: str = None,
):
    if output_csv_path:
        export_to_csv(models, all_data, output_csv_path)

    active_cats = set()
    for data in all_data:
        for counter in data.values():
            active_cats.update(counter.keys())

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, cfg, data in zip(axes, models, all_data):
        lengths = sorted(data.keys())

        for cat in ERROR_CATEGORIES:
            if cat not in active_cats:
                continue
            counts = [data[l].get(cat, 0) for l in lengths]
            if not any(c > 0 for c in counts):
                continue

            style = CAT_STYLE[cat]
            ax.plot(
                lengths, counts,
                marker=style["marker"],
                color=style["color"],
                linewidth=2.5,
                markersize=5,
                label=cat,
                zorder=3,
            )

        ax.set_title(cfg["name"], fontsize=10.5, fontweight="semibold", pad=6)
        ax.set_xlabel("Test Length", fontsize=10, fontweight="bold")
        ax.set_ylabel("Number of Errors", fontsize=10, fontweight="bold")
        ax.set_xticks(lengths)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        style_axes(ax)

    legend_handles = [
        mlines.Line2D(
            [], [],
            color=CAT_STYLE[cat]["color"],
            marker=CAT_STYLE[cat]["marker"],
            linewidth=2.5,
            markersize=5,
            label=cat,
        )
        for cat in ERROR_CATEGORIES if cat in active_cats
    ]
    _make_legend(fig, legend_handles, n_cols=3)

    fig.suptitle(
        "Error Category Evolution Across Test Lengths",
        fontsize=10.5, fontweight="semibold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0.15, 1, 0.95])

    if output_image_path:
        fig.savefig(output_image_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {output_image_path}")

    plt.show()


def plot_cross_scheme_comparison(
    models: list,
    all_data: list,
    output_image_path: str = None,
):
    SCHEME_COLORS = {
        "Naive": "#C44E52",
        "Signposts": "#2E8B57",
        "Signposts + Value-Change": "#2196F3",
    }
    SCHEME_MARKERS = ["o", "s", "D"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    all_lengths = set()
    legend_handles = []

    for idx, (cfg, data) in enumerate(zip(models, all_data)):
        lengths = sorted(data.keys())
        all_lengths.update(lengths)

        color  = SCHEME_COLORS[cfg["name"]]
        marker = SCHEME_MARKERS[idx % len(SCHEME_MARKERS)]
        scheme = cfg["scheme"]

        # Panel 1: Wrong Swap Boxes
        counts_swap = [data[l].get("Trace: Wrong Swap Boxes", 0) for l in lengths]
        for index,l in enumerate(lengths):
            counts_swap[index] += data[l].get("Trace: Wrong Index Loaded", 0)
        axes[0].plot(
            lengths, counts_swap,
            color=color, marker=marker, linewidth=2.5, markersize=5, zorder=3,
        )

        # Panel 2: Incorrect State Write / Wrong W/K Token
        cat_state = "Trace: Wrong W/K Token" if scheme == "delta_cot" else "Trace: Incorrect State Write"
        counts_state = [data[l].get(cat_state, 0) for l in lengths]
        axes[1].plot(
            lengths, counts_state,
            color=color, marker=marker, linewidth=2.5, markersize=5, zorder=3,
        )

        legend_handles.append(
            mlines.Line2D(
                [], [],
                color=color, marker=marker, linewidth=2.5, markersize=5,
                label=cfg["name"],
            )
        )

    sorted_lengths = sorted(all_lengths)

    titles = ["Wrong Copy Operation", "Incorrect State Produced"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10.5, fontweight="semibold", pad=6)
        ax.set_xlabel("Test Length", fontsize=10, fontweight="bold")
        ax.set_ylabel("Number of Errors", fontsize=10, fontweight="bold")
        ax.set_xticks(sorted_lengths)
        ax.tick_params(rotation=90)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        style_axes(ax)

    _make_legend(fig, legend_handles, n_cols=len(models))

    fig.tight_layout(rect=[0, 0.12, 1, 0.95])

    if output_image_path:
        fig.savefig(output_image_path, dpi=300, bbox_inches="tight")
        print(f"Comparison figure saved to: {output_image_path}")

    plt.show()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    MODELS = [
        {
            "name":   "Naive",
            "folder": "/path/to/run_folder",
            "scheme": "naive",
        },
        {
            "name":   "Signposts",
            "folder": "/path/to/run_folder",
            "scheme": "signposts",
        },
        {
            "name":   "Signposts + Value-Change",
            "folder": "/path/to/run_folder",
            "scheme": "delta_cot",
        },
    ]

    setup_style()

    print("Collecting data...")
    all_data = []
    for cfg in MODELS:
        print(f"\nAnalyzing '{cfg['name']}' ({cfg['scheme']}) ← {cfg['folder']}")
        all_data.append(analyze_folder(cfg["folder"], cfg["scheme"]))
    '''
    plot_error_evolution(
        MODELS, all_data,
        output_image_path="error_evolution.png",
        output_csv_path="error_evolution.csv",
    )'''

    plot_cross_scheme_comparison(
        MODELS, all_data,
        output_image_path="cross_scheme_comparison.png",
    )