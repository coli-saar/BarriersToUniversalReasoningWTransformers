import fire
import json
import lengthgen.evaluate_model as evaluate_model
import yaml

from lengthgen.paths import RESULTS_OUT_BASE_PATH

SELECTED_PATH = RESULTS_OUT_BASE_PATH / "selected_models"
OUT_PATH = RESULTS_OUT_BASE_PATH / "test_results"

def main(
    config: str,
    task: str = None,
    condition: str = None,
):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg["evaluation"]
    tasks_to_run = [task] if task else list(cfg["tasks"].keys())

    for t in tasks_to_run:
        task_cfg = cfg["tasks"][t]
    
        base_task_out = OUT_PATH / task_cfg['positional_encodings']
        
        conditions_to_run = [c for c in cfg["conditions"] if not condition or c["name"] == condition]

        for train_len in task_cfg["train_lengths"]:
            for cond in conditions_to_run:
                cond_name = cond["name"]
                sel_file = SELECTED_PATH / task_cfg['positional_encodings'] / f"{t}_len{train_len}_{cond_name}.json"
                if not sel_file.exists():
                    print(f"Skipping {sel_file} — run selection first")
                    continue

                with open(sel_file) as f:
                    selection = json.load(f)

                if not selection["top_runs"]:
                    print(f"No runs selected for {t} len{train_len} {cond_name}")
                    continue

                # Compute eval lengths from config
                el = task_cfg["eval_lengths"]
                min_eval = max(1, train_len + el["start_offset"])
                max_eval = int(train_len * el["ood_test_factor"])
                step = el["step"]

                print(f"\n{'='*50}")
                print(f"Task={t} train_len={train_len} condition={cond_name}")
                print(f"Eval range: {min_eval}..{max_eval} step {step}")
                print(f"Runs: {len(selection['top_runs'])}")

                # Merge task_kwargs with condition kwargs dynamically
                current_kwargs = task_cfg.get("task_kwargs", {}).copy()
                for k, v in cond.items():
                    if k != "name":
                        current_kwargs[k] = v
                
                repetitive = current_kwargs.get("repetitive", False)
                task_out = base_task_out / "repetitive" if repetitive else base_task_out
                task_out.mkdir(parents=True, exist_ok=True)

                for run in selection["top_runs"]:
                    print(f"Evaluating seed={run['seed']} {run['run_name']}")

                    evaluate_model.evaluate_model(
                        model_path = run["model_path"],
                        task = t,
                        min_len = min_eval,
                        max_len = max_eval,
                        step = step,
                        num_samples = eval_cfg["num_test_samples"],
                        seed = eval_cfg["test_seed"],
                        batch_size = eval_cfg["batch_size"],
                        save_to_disk = True,
                        out_path = task_out / t / f"len{train_len}" / cond_name,
                        task_kwargs = current_kwargs,
                    )

if __name__ == "__main__":
    fire.Fire(main)