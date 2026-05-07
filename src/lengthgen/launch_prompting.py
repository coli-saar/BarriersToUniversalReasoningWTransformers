import argparse
import json
import os
import subprocess
import sys
import yaml

from itertools import product
from lengthgen.paths import PROMPTING_FILE_PATH

def is_excluded(params, exclusions):
    for rule in exclusions:
        if all(params.get(k) == v for k, v in rule.items()):
            return True
    return False

def run_command(cmd, run_name):
    print(f"{run_name} Queued for execution...")
    try:
        result = subprocess.run(cmd, env=os.environ.copy(), capture_output=False, text=True)
        if result.returncode != 0:
            print(f"{run_name} Command failed with code {result.returncode}")
            return False
        print(f"{run_name} SUCCESS.")
        return True
    except Exception as e:
        print(f"[{run_name}] Unexpected error running command: {e}")
        return False

def construct_run_name(run_config, seed, model, hf_model, n_shots):
    task = run_config["task"]
    task_kwargs = run_config.get("task_kwargs", {})
    idx_str = "index_" if task_kwargs.get("include_indices", False) else ""
    delta_str = "delta_" if task_kwargs.get("delta_cot", False) else ""
    active_model = hf_model if hf_model else model
    model_str = active_model.split("/")[-1]
    backend_str = "hf_" if hf_model else ""
    min_len = run_config["min_test_len"]
    max_len = run_config["max_test_len"]

    return (
        f"{task}_{backend_str}{model_str}_"
        f"{idx_str}{delta_str}"
        f"{n_shots}shot_"
        f"len{min_len}-{max_len}_s{seed}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sweep = cfg.get("sweep", {})
    fixed = cfg.get("fixed", {})
    exclusions = cfg.get("exclude", [])

    keys = list(sweep.keys())
    values = list(sweep.values())

    commands_to_run = []

    for combination in product(*values):
        current_params = dict(zip(keys, combination))
        run_config = {**fixed, **current_params}

        if is_excluded(current_params, exclusions):
            print(f"Skipping excluded config: {current_params}")
            continue

        n_shots = run_config.pop("n_shots_list", 3)
        seed = run_config.pop("seeds", 42)
        model = run_config.pop("models", "meta-llama/Llama-3-70b-instruct")
        hf_model = run_config.pop("hf_models", None)

        run_name = construct_run_name(run_config, seed, model, hf_model, n_shots)

        cmd = [
            sys.executable, PROMPTING_FILE_PATH,
            "--task", str(run_config["task"]),
            "--model", str(model),
            "--min_test_len", str(run_config["min_test_len"]),
            "--max_test_len", str(run_config["max_test_len"]),
            "--n_shots", str(n_shots),
            "--num_eval_samples", str(run_config["num_eval_samples"]),
            "--seed", str(seed),
            "--max_new_tokens", str(run_config.get("max_new_tokens", 2048)),
            "--step_size", str(run_config.get("step_size", 5)),
            "--run_name", str(run_name),
        ]

        if hf_model:
            cmd.extend(["--hf_model", str(hf_model)])

        if "task_kwargs" in run_config:
            cmd.extend(["--task_kwargs", json.dumps(run_config["task_kwargs"])])

        commands_to_run.append((cmd, run_name))

    print(f"--- Discovered {len(commands_to_run)} valid configurations ---")

    for cmd, run_name in commands_to_run:
        try:
            success = run_command(cmd, run_name)
            if not success:
                print(f"{run_name} Failed to complete successfully.")
        except Exception as exc:
            print(f"{run_name} Generated an exception: {exc}")

if __name__ == "__main__":
    main()