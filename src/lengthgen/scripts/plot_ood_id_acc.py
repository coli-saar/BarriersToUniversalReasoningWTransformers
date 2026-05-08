import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

'''
Set the root directory where your task folders are located
Expected folder structure:

BASE_DIR/
├── {task}/ e.g. permutation/
│   └── {len_*}/ e.g. len_10/
│       └── {format}/ e.g. signpost/
│           ├── id.csv
│           └── ood.csv
└── ...
'''
BASE_DIR = Path("/home/oliver/Documents/lengthgen/appendix") 

ID_COLOR = "#2b5c8f"
OOD_COLOR = "#d95f02"

plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "font.size": 10,
    "legend.fontsize": 9,
})

def process_wandb_csv(csv_path):
    """Reads a W&B CSV, extracts the Step and accuracy columns, and renames them."""
    if not csv_path.exists():
        return None
        
    df = pd.read_csv(csv_path)
    
    step_col = [c for c in df.columns if c.strip().lower() == 'step']
    step_col = step_col[0] if step_col else df.columns[0]
    
    # Filter for columns that end strictly with 'accuracy' (ignoring MIN/MAX)
    # W&B often exports like "RunName - val/ood_accuracy"
    acc_cols = [c for c in df.columns if c.strip().endswith('accuracy')]
    
    # Keep only step and the valid accuracy columns
    df_clean = df[[step_col] + acc_cols].copy()
    
    # Rename columns to Seed 1, Seed 2, Seed 3...
    rename_dict = {step_col: "Step"}
    for i, col in enumerate(acc_cols):
        rename_dict[col] = f"Seed {i+1}"
        
    df_clean.rename(columns=rename_dict, inplace=True)
    return df_clean

# Discover all the configurations (Task -> Length -> Format)
plot_configs = []
for task_dir in [d for d in BASE_DIR.iterdir() if d.is_dir()]:
    for len_dir in [d for d in task_dir.iterdir() if d.is_dir() and 'len' in d.name]:
        for format_dir in [d for d in len_dir.iterdir() if d.is_dir()]:
            id_csv = format_dir / "id.csv"
            ood_csv = format_dir / "ood.csv"
            
            if id_csv.exists() and ood_csv.exists():
                plot_configs.append({
                    'task': task_dir.name.capitalize(),
                    'length': len_dir.name.replace('_', ' ').title(),
                    'format': format_dir.name.capitalize(),
                    'id_csv': id_csv,
                    'ood_csv': ood_csv
                })

# Sort configs so the grid is organized logically
plot_configs.sort(key=lambda x: (x['task'], x['length'], x['format']))

# --- Plotting ---
n_plots = len(plot_configs)
# Calculate grid size (e.g., 3 columns, auto-calculating rows)
cols = 3
rows = (n_plots + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), sharey=True)
# Flatten axes array for easy iteration
axes = axes.flatten() if n_plots > 1 else [axes]

sns.set_theme(style="whitegrid", rc={"axes.facecolor": (0, 0, 0, 0)})

for idx, config in enumerate(plot_configs):
    ax = axes[idx]
    
    df_id = process_wandb_csv(config['id_csv'])
    df_ood = process_wandb_csv(config['ood_csv'])
    
    # Plot ID curves for all seeds
    if df_id is not None:
        for seed_col in [c for c in df_id.columns if c.startswith('Seed')]:
            # Extract just the Step and this seed's column, then drop the NaNs
            clean_df = df_id[['Step', seed_col]].dropna()
            ax.plot(clean_df['Step'], clean_df[seed_col], color=ID_COLOR, alpha=0.6, linewidth=1.5)
            
    # Plot OOD curves for all seeds
    if df_ood is not None:
        for seed_col in [c for c in df_ood.columns if c.startswith('Seed')]:
            # Extract just the Step and this seed's column, then drop the NaNs
            clean_df = df_ood[['Step', seed_col]].dropna()
            ax.plot(clean_df['Step'], clean_df[seed_col], color=OOD_COLOR, alpha=0.6, linewidth=1.5)

    # Clean up subplot titles and axes
    ax.set_title(f"{config['task']} | {config['length']} | {config['format']}")
    ax.set_xlabel("Training Steps")
    if idx % cols == 0:
        ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)

# Hide any empty subplots if the grid isn't perfectly filled
for idx in range(n_plots, len(axes)):
    fig.delaxes(axes[idx])

# Create custom legend entries for the whole figure
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color=ID_COLOR, lw=2, alpha=0.8),
    Line2D([0], [0], color=OOD_COLOR, lw=2, alpha=0.8)
]
fig.legend(custom_lines, ['In-Domain (ID) Accuracy', 'Out-Of-Domain (OOD) Accuracy'], 
           loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

sns.despine(fig=fig)
plt.tight_layout()

# Save the plot
output_file = "validation_grid.pdf"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Successfully generated grid with {n_plots} subplots -> {output_file}")