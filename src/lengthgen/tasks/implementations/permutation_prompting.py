import hashlib
import os
import random
import yaml

from datasets import Dataset, disable_caching
from lengthgen.constants import PROMPTING_TRACE_TOKEN, FINAL_ANSWER_PROMPTING_TOKEN, END_OF_TEXT_PROMPTING_TOKEN
from pathlib import Path
from tqdm import tqdm

OBJECT_POOL = [
    "Cat", "Dog", "Apple", "Book", "Hat",
]
INDEX_TO_CHAR = {i: chr(ord("A") + i - 1) for i in range(1, 6)}

def get_idx_str(i: int):
    return f"{i}"

def char(i: int):
    return INDEX_TO_CHAR[i]

def build_op_string(box1: int, box2: int, curr_id: str, include_indices: bool):
    label = f"{curr_id} " if include_indices else ""
    return f"{label}swap {char(box1)} {char(box2)} ."

def build_trace_step(
    box1: int, box2: int, val_i: str, val_j: str,
    state: list[str], curr_id: str,
    include_indices: bool
):
    indexed_state = " ".join(f"{char(k+1)}={obj}" for k, obj in enumerate(state))
    label = f"{curr_id} " if include_indices else ""

    if include_indices:
        return (
            f"step {label}"
            f"swap contents {char(box1)}={state[box2-1]} {char(box2)}={state[box1-1]}. "
            f"new state: {indexed_state} . "
        )
    else:
        return (
            f"swap contents {char(box1)}={state[box2-1]} {char(box2)}={state[box1-1]}. "
            f"write {indexed_state} . "
        )

def _permutation_generator(
    num_samples: int,
    min_length: int,
    max_length: int,
    seed: int,
    num_objects: int,
    include_indices: bool,
    repetitive: bool,
):
    rng = random.Random(seed)
    seen = set()
    for _ in tqdm(range(num_samples), desc="Creating permutation dataset"):
        for _ in range(10):
            state = rng.sample(OBJECT_POOL, num_objects)
            init_desc = "init " + " ".join(f"{char(i+1)}={obj}" for i, obj in enumerate(state)) + " "
            length = rng.randint(min_length, max_length)

            start_idx = 1
            # Need (max_len + 1) * 2 indices: max_len digits + 1 sentinel per number
            chain_ids = list(range(start_idx, start_idx + length + 1))

            get_curr_id = lambda step: get_idx_str(chain_ids[step])

            ops, trace_steps = [], []
            prev_i = prev_j = None

            for step in range(length):
                if repetitive and step > 0 and rng.random() < 0.9:
                    i, j = prev_i, prev_j
                else:
                    i, j = rng.sample(range(num_objects), 2)

                val_i, val_j = state[i], state[j]
                state[i], state[j] = state[j], state[i]

                box1, box2 = i + 1, j + 1
                curr_id = get_curr_id(step)

                ops.append(build_op_string(box1, box2, curr_id, include_indices))
                trace_steps.append(build_trace_step(
                    box1, box2, val_i, val_j, state, curr_id, include_indices
                ))
                prev_i, prev_j = i, j

            trace_str = "".join(trace_steps)

            trace_str += "end "
            indexed_state = " ".join(f"{char(k+1)}={obj}" for k, obj in enumerate(state))
            answer_str = "".join(indexed_state)
            ops_block = " ".join(ops) + " end ."
            prompt_str = f"{init_desc}operation {ops_block}"

            full_text = f"{prompt_str} {PROMPTING_TRACE_TOKEN} {trace_str} final state: {FINAL_ANSWER_PROMPTING_TOKEN} {answer_str}"
            full_text += f" {END_OF_TEXT_PROMPTING_TOKEN}"
            h = int.from_bytes(hashlib.blake2b(full_text.encode(), digest_size=16).digest(), "little")
            if h not in seen:
                seen.add(h)
                break
        yield {"text": full_text}

def generate_permutation_prompting_dataset(
    num_samples: int,
    min_length: int,
    max_length: int,
    seed: int,
    num_objects: int = 5,
    include_indices: bool = True,
    save_to_path: str = None,
    repetitive: bool = False,
):
    disable_caching()

    dataset = Dataset.from_generator(
        _permutation_generator,
        gen_kwargs={
            "num_samples": num_samples,
            "min_length": min_length,
            "max_length": max_length,
            "seed": seed,
            "num_objects": num_objects,
            "include_indices": include_indices,
            "repetitive": repetitive,
        },
        keep_in_memory=True
    )

    if save_to_path:
        save_path_str = str(save_to_path)
        print(f"Saving to: {save_path_str}")
        parent_dir = os.path.dirname(save_path_str)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        dataset.save_to_disk(save_path_str)

    return dataset

def update_yaml_prompts(
    yaml_path: str,
    num_shots: int = 5,
    min_len: int = 3,
    max_len: int = 6,
    seed: int = 42
):
    path = Path(yaml_path)
    
    if path.exists():
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        print("WARNING. Yaml path not found!")
        return

    print(f"Generating {num_shots} baseline examples...")
    baseline_ds = generate_permutation_prompting_dataset(
        num_samples=num_shots,
        min_length=min_len,
        max_length=max_len,
        seed=seed,
        include_indices=False,
    )
    
    data["examples_base"] = [{"text": ex["text"]} for ex in baseline_ds]

    print(f"Generating {num_shots} index examples...")

    index_ds = generate_permutation_prompting_dataset(
        num_samples=num_shots,
        min_length=min_len,
        max_length=max_len,
        seed=seed,
        include_indices=True,
    )
    
    data["examples_index"] = [{"text": ex["text"]} for ex in index_ds]

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
    print(f"\nSuccessfully updated {yaml_path}!")
    print(f"Injected {num_shots} baseline shots and {num_shots} delta shots.")

if __name__ == "__main__":
    ds_exp = generate_permutation_prompting_dataset(
        num_samples=3, min_length=4, max_length=4, seed=44, num_objects=5, include_indices=False
    )
    for ex in ds_exp:
        print(ex["text"])
        print("\n")
    update_yaml_prompts("/path/to/permutation_prompting.yaml")
