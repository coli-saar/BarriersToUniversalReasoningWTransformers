import fire
import json
import pandas as pd
import random
import re
import torch

from lengthgen.tasks import registry as task_registry
from lengthgen.tasks.loader import load_split
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from lengthgen.paths import RESULTS_OUT_BASE_PATH
from lengthgen.constants import TRACE_TOKEN, FINAL_ANSWER_TOKEN, PADDING_TOKEN, END_OF_TEXT_TOKEN

DEFAULT_OUT_LOCATION = RESULTS_OUT_BASE_PATH
DEFAULT_OUT_LOCATION.mkdir(parents=True, exist_ok=True)

def normalize(s):
    if not isinstance(s, str):
        return ""
    s = s.replace(PADDING_TOKEN, "").replace(END_OF_TEXT_TOKEN, "")
    s = re.sub(r'<[^>]+>', '', s)
    s = s.replace('$', '')
    return s.replace(" ", "")

def evaluate_model(
    model_path,
    task="permutation",
    min_len=15,
    max_len=30,
    step=1,
    num_samples=100,
    seed=4096,
    save_to_disk=True,
    batch_size=32,
    starting_aid=0,
    task_kwargs="{}",
    max_new_tokens=3200,
    out_path=None,
):
    print(f"Evaluating model: {model_path}")
    random.seed(seed)
    
    if isinstance(task_kwargs, str):
        task_kwargs = json.loads(task_kwargs)
    elif not isinstance(task_kwargs, dict):
        raise ValueError(f"Invalid task_kwargs. Expected dict or str, got {type(task_kwargs)}")

    repetitive_str = "_repetitive" if task_kwargs.get("repetitive", False) else ""

    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    current_vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > current_vocab_size:
        print(f"WARNING: Tokenizer ({len(tokenizer)}) > Model ({current_vocab_size}). Resizing...")
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    model_name = Path(model_path).name

    if out_path:
        out_dir = Path(out_path) / model_name
    else:
        out_dir = Path(DEFAULT_OUT_LOCATION / model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_lengths = list(range(min_len, max_len + 1, step))
    print(f"Testing lengths: {test_lengths}")
    
    results = {}
    extra_results = {}
    
    for length in test_lengths:
        print(f"Testing length {length}")
        test_dataset = load_split(
            task=task,
            split="val", # we can reuse val split as test here. Just need to make sure we use an unseen ran seed
            num_samples=num_samples, 
            min_length=length,
            max_length=length,
            seed=seed + length,
            **task_kwargs,
        )
        
        correct = 0
        checkpoint_correct = 0
        results_log = []
        
        prompts = []
        targets = []
        extras = [] # Store ground truth metadata (e.g. checkpoint for multiplication)
        try:
            for i, item in enumerate(test_dataset):
                text = item["text"]
                
                if TRACE_TOKEN in text:
                    prompt_part = text.split(TRACE_TOKEN)[0] + TRACE_TOKEN
                    target_answer = text.split(f"{FINAL_ANSWER_TOKEN} ")[1].strip().replace(END_OF_TEXT_TOKEN, "")
                    if starting_aid > 0:
                        trace_remainder = text.split(TRACE_TOKEN)[1]
                        # Encode without special tokens so we only get the raw text tokens
                        trace_tokens = tokenizer.encode(trace_remainder, add_special_tokens=False)
                        aid_text = tokenizer.decode(trace_tokens[:starting_aid])
                        prompt_part += " " + aid_text
                    # Check for Multiplication Grid in Ground Truth
                    expected_checkpoint = None
                    if "checkpoint:" in text:
                        try:
                            expected_checkpoint = text.split("checkpoint")[1].split("||")[0].strip()
                        except IndexError:
                            pass

                    prompts.append(prompt_part)
                    targets.append(target_answer)
                    extras.append({"checkpoint": expected_checkpoint})
                else:
                    print(f"Warning: Separator not found in sample {i}")
                    prompts.append(None)
                    targets.append(None)
                    extras.append(None)
    
            num_batches = (len(prompts) + batch_size - 1) // batch_size
            for batch_idx in tqdm(range(num_batches), desc=f"Length {length}"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(prompts))

                batch_prompts = prompts[start_idx:end_idx]
                batch_targets = targets[start_idx:end_idx]
                batch_extras = extras[start_idx:end_idx]

                valid_indices = [i for i, p in enumerate(batch_prompts) if p is not None]
                if not valid_indices:
                    continue

                valid_prompts = [batch_prompts[i] for i in valid_indices]
                valid_targets = [batch_targets[i] for i in valid_indices]
                valid_extras = [batch_extras[i] for i in valid_indices]

                inputs = tokenizer(
                    valid_prompts, 
                    return_tensors="pt", 
                    padding=True,
                ).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id
                    )

                full_outs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for local_idx, (full_out, target_answer, extra) in enumerate(zip(full_outs, valid_targets, valid_extras)):
                    global_idx = start_idx + valid_indices[local_idx]
                    target_answer = target_answer.strip()
                    try:
                        prediction = full_out.split(f"{FINAL_ANSWER_TOKEN}")[1].strip()
                    except IndexError:
                        prediction = "ERROR"

                    is_checkpoint_correct = None
                    if extra and extra["checkpoint"] is not None:
                        try:
                            pred_grid = full_out.split("checkpoint")[1].split("||")[0].strip()
                            is_checkpoint_correct = (normalize(pred_grid) == normalize(extra["checkpoint"]))
                        except IndexError:
                            is_checkpoint_correct = False

                    if task == task_registry.TaskType.ADDITION or task == task_registry.TaskType.MULTIPLICATION:
                        try:
                            prediction = normalize(prediction)
                            target_answer = normalize(target_answer)
                        except Exception as e: # in case there is no valid prediction (i.e. None)
                            pass
                    is_correct = (prediction == target_answer)
                    
                    if is_correct:
                        correct += 1
                    if is_checkpoint_correct:
                        checkpoint_correct += 1

                    if save_to_disk:
                        results_log.append({
                            "id": global_idx,
                            "correct": is_correct,
                            "checkpoint_correct": is_checkpoint_correct,
                            "target": target_answer,
                            "prediction": prediction,
                            "full_output": full_out,
                            "prompt": valid_prompts[local_idx]
                        })
        except Exception as e:
            print(e)
        acc = correct / num_samples
        results[length] = acc
        print(f"Length {length}: {acc:.2%} ({correct}/{num_samples})")
        if task == task_registry.TaskType.MULTIPLICATION:
            checkpoint_acc = checkpoint_correct / num_samples
            extra_results[length] = checkpoint_acc
            print(f"Length {length} - checkpoints: {checkpoint_acc:.2%} ({checkpoint_correct}/{num_samples})")

        if save_to_disk:
            filename = f"eval_{task}_len{length}{repetitive_str}.json"
            save_path = out_dir / filename
            with open(save_path, "w") as f:
                json.dump(results_log, f, indent=2)
            print(f"Debug logs saved to: {save_path}")

    print("Final Results:")
    if task == task_registry.TaskType.MULTIPLICATION:
        data = []
        for length in results.keys():
            data.append({
                "Length": length,
                "Accuracy": results[length],
                "Checkpoint Accuracy": extra_results.get(length, 0.0)
            })
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(list(results.items()), columns=["Length", "Accuracy"])
    print(df)
    
    save_path = out_dir / f"full_results_{task}_{model_name}{repetitive_str}.csv"
    df.to_csv(save_path, index=False)
    
    print(f"Saved results to {save_path}")

if __name__ == "__main__":
    fire.Fire(evaluate_model)