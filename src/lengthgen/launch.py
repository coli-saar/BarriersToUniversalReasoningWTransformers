import argparse
import glob
import json
import os
import socket
import subprocess
import sys
import yaml
from itertools import product
from lengthgen.tasks import registry as task_registry

from lengthgen.paths import TRAIN_FILE_PATH, EVAL_FILE_PATH, DATA_BASE_PATH, HF_CACHE_LOCATION

def is_excluded(params, exclusions):
    for rule in exclusions:
        if all(params.get(k) == v for k, v in rule.items()):
            return True
    return False

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def cleanup(pattern):
    matches = glob.glob(pattern)
    if matches:
        subprocess.run(["rm", "-rf"] + matches)
    else:
        print(f"No files matched: {pattern}")

def run_command(cmd, env=None):
    print(f"Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, env=env if env else os.environ.copy())
        if result.returncode != 0:
            print(f"Error: Command failed with code {result.returncode}")
            sys.exit(result.returncode)
    except Exception as e:
        print(f"Unexpected error running command: {e}")
        sys.exit(1)

def construct_run_name(run_config, seed, length, wdecay, lr, pos_dropout_prob, rep_ratio,target_bsize):
    task = run_config['task']
    task_kwargs = run_config.get("task_kwargs", {})
    idx_str = "index_" if task_kwargs.get("include_indices", False) else ""
    delta_str = "delta_" if task_kwargs.get("delta_cot", False) else ""
    goto_str = "goto_" if task_kwargs.get("use_goto", False) else ""
    adversarial_digit_ratio = task_kwargs.get("adversarial_digit_ratio", 0.0)
    adversarial_digit_ratio_str = f"adr_{str(adversarial_digit_ratio).replace('.', '')}" if adversarial_digit_ratio > 0.0 else ""
    pos_str = "ran_pos_" if str(run_config["random_pos"]) == "True" else ""
    
    num_objects = run_config.get("task_kwargs", {}).get("num_objects", 5)
    dropout_str = str(pos_dropout_prob).replace(".", "")
    rep_ratio_str = str(rep_ratio).replace(".", "")
    obj_str = str(num_objects) if task == task_registry.TaskType.PERMUTATION else ""
    wdecay = str(wdecay).replace(".", "")
    lr = str(lr).replace(".", "")

    return (
        f"{run_config['run_name_prefix']}{run_config['task']}{obj_str}_"
        f"{run_config['positional_encodings']}_{pos_str}{idx_str}{delta_str}{goto_str}"
        f"len{length}_wd{wdecay}_do{dropout_str}_rr{rep_ratio_str}_lr{lr}_tbs{target_bsize}{adversarial_digit_ratio_str}_s{seed}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--num_gpus", type=int, required=True, help="Num of GPUs to run the task")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sweep = cfg.get("sweep", {})
    fixed = cfg.get("fixed", {})
    num_gpus = args.num_gpus

    master_port = find_free_port() if cfg["resources"]["master_port"] == "auto" else cfg["resources"]["master_port"]
    
    print(f"Detected {num_gpus} GPUs. Using Port {master_port}.")

    keys = list(sweep.keys())
    values = list(sweep.values())
    exclusions = cfg.get("exclude", [])
    
    for combination in product(*values):
        current_params = dict(zip(keys, combination))
        run_config = {**fixed, **current_params}

        if is_excluded(current_params, exclusions):
            print(f"Skipping excluded config: {current_params}")
            continue

        resume_from_path = run_config.pop("resume_from_path", None)
        seed = run_config.pop("seeds", 42)
        lr = run_config.pop("learning_rates", 3e-4)
        length = run_config.pop("lengths", 30)
        wdecay = run_config.pop("wdecays", 0.1)
        warmup_ratio = run_config.pop("warmup_ratios", 0.1)
        rep_ratio = run_config.pop("rep_ratios", 0.1)
        task = str(run_config["task"])
        minimum_lr = str(run_config.get("minimum_lr", 0.1))
        num_train_samples = str(run_config["num_train_samples_list"])
        pos_dropout_prob = str(run_config["pos_dropout_probs"])
        target_bsize = str(run_config["target_bsizes"])
        
        abort_step = run_config.pop("abort_step", None)
        abort_eval_cutoff = run_config.pop("abort_eval_cutoff", None)

        base_run_name = construct_run_name(run_config,seed,length,wdecay,lr,pos_dropout_prob,rep_ratio,target_bsize)
        
        # If resuming, train.py will append "_continued" internally. 
        # We need to track the final name for the evaluator.
        is_resuming = bool(resume_from_path and str(resume_from_path).lower() != "none")
        final_run_name = f"{base_run_name}_continued" if is_resuming else base_run_name
        print(f"--- Starting Run: {final_run_name} ---")

        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            f"--rdzv_endpoint=localhost:{master_port}",
            str(TRAIN_FILE_PATH),
            "--task", task,
            "--train_len", str(length),
            "--batch_size", str(run_config["batch_size"]),
            "--target_bsize", str(target_bsize),
            "--num_train_samples", str(num_train_samples),
            "--lr", str(lr),
            "--minimum_lr", str(minimum_lr),
            "--seed", str(seed),
            "--warmup_ratio", str(warmup_ratio),
            "--output_dir", str(run_config["output_dir"]),
            "--wdecay", str(wdecay),
            "--positional_encodings", str(run_config["positional_encodings"]),
            "--run_name", str(base_run_name)
        ]
        if resume_from_path:
            cmd.extend(["--resume_from_path", str(resume_from_path)])
        if abort_step is not None:
            cmd.extend(["--abort_step", str(abort_step)])
        if abort_eval_cutoff is not None:
            cmd.extend(["--abort_eval_cutoff", str(abort_eval_cutoff)])

        bool_keys = ["random_pos", "best_ckpt"]
        for k in bool_keys:
            if k in run_config:
                cmd.extend([f"--{k}", str(run_config[k])])

        if "task_kwargs" in run_config:
            task_str = json.dumps(run_config["task_kwargs"])
            cmd.extend(["--task_kwargs", task_str])
        print(f"Train file path: {TRAIN_FILE_PATH}")
        print(f"Running: {' '.join(cmd)}")
        run_command(cmd)

        model_path = os.path.join(run_config["output_dir"], final_run_name)

        print(f"Evaluating Model at: {model_path}")

        # Define Min/Max Eval Lengths
        min_len = int(length) - 10
        max_len = int(length) + 29 # might need to do this dynamically based on train len...
        if min_len < 1: min_len = 1

        eval_cmd_base = [
            sys.executable, str(EVAL_FILE_PATH),
            "--task", str(run_config["task"]),
            "--model_path", model_path,
            "--min_len", str(min_len),
            "--max_len", str(max_len),
        ]
        if "task_kwargs" in run_config:
            eval_cmd_base.extend(["--task_kwargs", json.dumps(run_config["task_kwargs"])])
        
        # optionally run eval at the end of training
        #run_command(eval_cmd_base)

        # cleanup
        cleanup(f"{DATA_BASE_PATH}/*{final_run_name}*")
        cleanup(f"{HF_CACHE_LOCATION}/*{final_run_name}*")

if __name__ == "__main__":
    main()