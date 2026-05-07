import fire
import json
import os
import pandas as pd
import random
import re
import time
import yaml

from lengthgen.constants import PROMPTING_TRACE_TOKEN, END_OF_TEXT_PROMPTING_TOKEN, FINAL_ANSWER_PROMPTING_TOKEN
from lengthgen.paths import RESULTS_OUT_BASE_PATH, PROMPTS_BASE_PATH
from lengthgen.tasks import registry as tasks_registry
from lengthgen.tasks.loader import load_split
from pathlib import Path
from together import Together
from tqdm import tqdm
from typing import Callable

def get_client():
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    return Together(api_key=api_key)

def make_together_query_fn(
    client,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    retries: int = 3,
    retry_delay: float = 5.0,
):
    def query_fn(prompt: str) -> str:
        for attempt in range(retries):
            try:
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=[END_OF_TEXT_PROMPTING_TOKEN, "finished"],
                )
                return response.choices[0].text
            except Exception as e:
                if attempt < retries - 1:
                    print(f"API error (attempt {attempt+1}/{retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"API error after {retries} attempts: {e}")
                    return ""
        return ""
    return query_fn

def make_hf_query_fn(
    model_name: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError("transformers and torch are required for local HF inference.")

    print(f"Loading HuggingFace model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")

    stop_string = END_OF_TEXT_PROMPTING_TOKEN

    def query_fn(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(new_ids, skip_special_tokens=True)

        # Honour the same stop string as the Together API
        if stop_string in generated:
            generated = generated[:generated.index(stop_string)]
        return generated

    return query_fn

def load_prompt_file(task: str):
    yaml_path = PROMPTS_BASE_PATH / f"{task}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def build_few_shot_prompt(
    prompt_file: dict,
    shot_examples: list[str],
    test_prompt: str,
    mode: str,
):
    instruction = prompt_file[f"instruction_{mode}"]
    parts = [instruction]
    parts.append("\n")
    if len(shot_examples) > 0:
        parts.append("Examples: \n")
    for ex in shot_examples:
        parts.append(ex['text'].strip())
        parts.append("\n\n")
    parts.append("Your task: \n")
    parts.append(test_prompt)
    return "".join(parts)

def extract_prompt(full_text: str):
    prompt = full_text.split(PROMPTING_TRACE_TOKEN)[0]
    prompt += f" {PROMPTING_TRACE_TOKEN}"
    return prompt

def extract_answer(text: str):
    # Normalize literal \n (from un-decoded JSON strings) into real newlines
    normalized = text.replace('\n', ' ')
    
    match = re.search(rf'(?i){FINAL_ANSWER_PROMPTING_TOKEN}\s*:?\s*([^\n<]+)', normalized)
    if match:
        raw = match.group(1).replace(END_OF_TEXT_PROMPTING_TOKEN, "")
        return " ".join(raw.split())
    return None

def parse_answer_to_dict(answer_str: str) -> dict:
    if not answer_str:
        return {}
    matches = re.findall(r'([a-zA-Z_]\w*)\s*=\s*(\d+)', answer_str)
    return {k: int(v) for k, v in matches}

from lengthgen.tasks import registry as tasks_registry

def score(pred_str: str, ground_truth_str: str, task: str):
    if pred_str is None:
        return False

    if task == tasks_registry.TaskType.PERMUTATION_PROMPTING:
        pred_values = _extract_permutation_values(pred_str)
        gt_values = _extract_permutation_values(ground_truth_str)
        return gt_values == pred_values and len(gt_values) > 0

    return pred_str == ground_truth_str

def _extract_permutation_values(s: str):
    matches = re.findall(r'([A-E])=(\w+)', s)
    if matches:
        return [v for _, v in sorted(matches, key=lambda x: x[0])]
    # Fallback: treat as space-separated token list
    return s.strip().split()

def evaluate(
    query_fn: Callable[[str], str],
    prompt_file,
    shot_examples: list[str],
    eval_dataset,
    task: str,
    mode: str,
    debug_samples: int = 5,
    out_path: str = None,
):
    correct = 0
    total = 0
    debug_printed = 0
    raw_outputs = []
    sample_id = 0

    for example in tqdm(eval_dataset, desc="Evaluating"):
        full_text = example["text"]
        ground_truth_str = extract_answer(full_text)
        if ground_truth_str is None:
            print("WARNING. No ground truth found")
            continue

        test_prompt = extract_prompt(full_text)
        prompt = build_few_shot_prompt(prompt_file, shot_examples, test_prompt, mode)

        generated = query_fn(prompt)
        pred_str = extract_answer(generated)

        is_correct = score(pred_str, ground_truth_str, task)

        if is_correct:
            correct += 1
        raw_outputs.append({
            "prompt": prompt,
            "generated": generated,
            "pred": pred_str,
            "ground_truth": ground_truth_str,
            "correct": is_correct,
            "id": sample_id,
        })

        if debug_printed < debug_samples:
            print(f"\n--- Debug sample {debug_printed + 1} ---")
            print(f"GT: {ground_truth_str}")
            print(f"Pred: {pred_str}")
            print(f"Generated: {generated[:200]}")
            debug_printed += 1
        total += 1
        sample_id += 1

    if out_path:
        with open(out_path, "w") as f:
            json.dump(raw_outputs, f, indent=2)

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}

def _normalize_bools(d: dict):
    def parse(v):
        if isinstance(v, bool): return v
        if isinstance(v, str) and v.lower() == 'false': return False
        if isinstance(v, str) and v.lower() == 'true': return True
        return v
    return {k: parse(v) for k, v in d.items()}

def get_prompt_mode(include_indices: bool, delta_cot: bool):
    if include_indices:
        return "index"
    if delta_cot:
        return "delta"
    return "base"

def main(
        task: str,
        model: str = "meta-llama/Llama-3-70b-instruct",
        hf_model: str = None,
        min_test_len: int = 10,
        max_test_len: int = 20,
        step_size: int = 5,
        num_eval_samples: int = 100,
        seed: int = 42,
        max_new_tokens: int = 2048,
        n_shots: int = 3,
        run_name: str = None,
        task_kwargs: str = '{}'
):
    random.seed(seed)
    config_dict = task_kwargs if isinstance(task_kwargs, dict) else json.loads(task_kwargs)
   
    config_dict = _normalize_bools(config_dict)
    include_indices = config_dict.get("include_indices", False)
    delta_cot = config_dict.get("delta_cot", False)
    mode = get_prompt_mode(include_indices, delta_cot)
    
    if hf_model:
        active_model = hf_model
        query_fn = make_hf_query_fn(hf_model, max_tokens=max_new_tokens)
    else:
        active_model = model
        client = get_client()
        query_fn = make_together_query_fn(client, model, max_tokens=max_new_tokens)

    if not run_name:
        raise ValueError("No valid run name provided")

    output_base_folder = RESULTS_OUT_BASE_PATH / "prompting" / task / Path(active_model).name / run_name
    output_base_folder.mkdir(parents=True, exist_ok=True)
    
    prompt_file = load_prompt_file(task)
    shot_examples = prompt_file[f"examples_{mode}"][:n_shots]

    config = {
        "task": task,
        "model": active_model,
        "backend": "hf" if hf_model else "together",
        "min_test_len": min_test_len,
        "max_test_len": max_test_len,
        "n_shots": n_shots,
        "seed": seed,
        **config_dict,
    }

    config_path = output_base_folder / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    aggregate_results = {}

    for test_len in range(min_test_len, max_test_len + 1, step_size):
        print(f"Evaluating {n_shots}-shot examples at test_len={test_len}...")
        result_output_path = output_base_folder / f"results_{run_name}_len_{test_len}.json"

        eval_dataset = load_split(
            task, "val",
            num_samples=num_eval_samples,
            min_length=test_len,
            max_length=test_len,
            seed=seed,
            **config_dict, 
        )

        results_dict = evaluate(
            query_fn,
            prompt_file,
            shot_examples,
            eval_dataset,
            task,
            mode=mode,
            out_path=result_output_path,
        )
        print(f"Accuracy: {results_dict['accuracy']:.2%} ({results_dict['correct']}/{results_dict['total']})")
        aggregate_results[test_len] = results_dict["accuracy"]

    print("\nFinal Results:")
    df = pd.DataFrame(list(aggregate_results.items()), columns=["Length", "Accuracy"])
    print(df)
    
    save_path = output_base_folder / f"full_results_{run_name}.csv"
    df.to_csv(save_path, index=False)
    print(f"Saved aggregated results to {save_path}")

if __name__ == "__main__":
    fire.Fire(main)