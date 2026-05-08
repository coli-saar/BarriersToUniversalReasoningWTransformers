import fire
import json
import pandas as pd
import wandb
import yaml

from lengthgen.paths import RESULTS_OUT_BASE_PATH
from pathlib import Path

SELECTED_MODELS_PATH = RESULTS_OUT_BASE_PATH / "selected_models"

def select_runs(task, train_len, condition, config, top_n=3):
    api = wandb.Api()
    project = config["tasks"][task]["wandb_project"]
    best_ckpt = config["selection"].get("best_ckpt", False)

    filters = {
        "state": "finished",
        "config.train_len": train_len,
    }

    # Dynamically inject condition properties as filters
    for k, v in condition.items():
        if k != "name":
            filters[f"config.task_kwargs.{k}"] = v

    runs = api.runs(
        f"{config['wandb_entity']}/{project}",
        filters=filters,
    )
    print(f"Found {len(runs)} matching runs")
    records = []
    for run in runs:
        # load full run
        full_run = api.run(f"{config['wandb_entity']}/{project}/{run.id}")
        if best_ckpt:
            history = full_run.history(keys=["eval/ood_accuracy", "eval/id_accuracy"])

            if history.empty or "eval/ood_accuracy" not in history.columns:
                print(f"Skipping {full_run.name}: no ood_accuracy in history")
                continue

            # Best OOD acc achieved at any checkpoint
            best_idx = history["eval/ood_accuracy"].idxmax()
            best_ood = history.loc[best_idx, "eval/ood_accuracy"]
            best_id = history.loc[best_idx, "eval/id_accuracy"] if "eval/id_accuracy" in history.columns else None
        else:
            best_id = full_run.summary.get("eval/id_accuracy")
            best_ood = full_run.summary.get("eval/ood_accuracy")

        output_dir = full_run.config.get("output_dir")
        run_name = full_run.config.get("run_name")
        records.append({
                "run_id": full_run.id,
                "run_name": full_run.name,
                "model_path": str(Path(output_dir) / run_name) if output_dir and run_name else None,
                "seed": full_run.config.get("seed"),
                "lr": full_run.config.get("learning_rate"),
                "wdecay": full_run.config.get("weight_decay"),
                "id_acc": best_id,
                "ood_acc": best_ood,
            })

    if not records:
        print(f"WARNING: no runs found for {task} len{train_len}"
              f"{condition['name']} with filters {filters}")
        return []

    df = pd.DataFrame(records).dropna(subset=["id_acc", "ood_acc"])

    df_sorted = df.sort_values("ood_acc", ascending=False)
    df_best_per_seed = df_sorted.drop_duplicates(subset=["seed"], keep="first")
    top = df_best_per_seed.nlargest(top_n, "ood_acc")

    return top.to_dict("records")

def main(
    config: str,
    task: str = None,
    condition: str = None,
):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    tasks_to_run = [task] if task else list(cfg["tasks"].keys())
    conditions = (
        [c for c in cfg["conditions"] if c["name"] == condition]
        if condition else cfg["conditions"]
    )

    for t in tasks_to_run:
        task_cfg = cfg["tasks"][t]
        out_dir = SELECTED_MODELS_PATH / task_cfg['positional_encodings']
        out_dir.mkdir(parents=True, exist_ok=True)
        for train_len in task_cfg["train_lengths"]:
            for cond in conditions:
                print(f"Selecting: {t} len{train_len} {cond['name']}...")

                runs = select_runs(
                    task=t,
                    train_len=train_len,
                    condition=cond,
                    config=cfg,
                    top_n=cfg["selection"]["top_n"],
                )

                out = {
                    "task": t,
                    "train_len": train_len,
                    "condition": cond["name"],
                    "top_runs": runs,
                }

                fname = f"{t}_len{train_len}_{cond['name']}.json"
                with open(out_dir / fname, "w") as f:
                    json.dump(out, f, indent=2)

                print(f"Selected {len(runs)} runs")
                for r in runs:
                    print(f"seed={r['seed']} "
                          f"id_acc={r['id_acc']:.3f} "
                          f"ood_acc={r['ood_acc']:.3f} "
                          f"{r['run_name']}")

if __name__ == "__main__":
    fire.Fire(main)