import hashlib
import os
import random

from lengthgen.constants import TRACE_TOKEN, FINAL_ANSWER_TOKEN, END_OF_TEXT_TOKEN
from datasets import Dataset, disable_caching
from tqdm import tqdm

OBJECT_POOL = [
    "Cat", "Dog"
]
INDEX_TO_CHAR = {i: chr(ord("A") + i - 1) for i in range(1, 6)}
# probability for which an operation is repeated (if the sample is selected as repetitive)
REPETITION_PROBABILITY = 0.9

def get_idx_str(i: int):
    return f"<{i}>"

def char(i: int):
    return INDEX_TO_CHAR[i]

def build_op_string(box1: int, box2: int, curr_id: str, include_indices: bool):
    label = f"{curr_id} " if include_indices else ""
    return f"{label}swap {char(box1)} {char(box2)} {label}."

def build_trace_step(
    box1: int, box2: int, val_i: str, val_j: str,
    state: list[str], curr_id: str,
    delta_cot: bool, include_indices: bool
):
    label = f"{curr_id} " if include_indices else ""

    if delta_cot:
        action_tokens = []

        if val_i != val_j:
            action_tokens.append(f"W_{char(box1)} {val_i}_{val_j}")
            action_tokens.append(f"W_{char(box2)} {val_j}_{val_i}")
        else:
            action_tokens.append(f"K_{char(box1)}")
            action_tokens.append(f"K_{char(box2)}")
                
        action_str = " ".join(action_tokens)
        return f"load {curr_id} . {label}swap {char(box1)} {char(box2)} {action_str} "
    else:
        indexed_state = " ".join(f"{char(k+1)} {obj}" for k, obj in enumerate(state))
        if include_indices:
            return f"load {curr_id} . {label}swap {char(box1)} {char(box2)} write {indexed_state} "
        else:
            return f"swap {char(box1)} {char(box2)} write {indexed_state} . "

def _permutation_generator(
    num_samples: int,
    min_length: int,
    max_length: int,
    seed: int,
    num_objects: int,
    include_indices: bool,
    delta_cot: bool,
    repetitive_mix: bool,
    repetitive_mix_ratio: float,
    repetitive: bool,
    max_index: int
):
    rng = random.Random(seed)
    seen = set()
    
    if repetitive_mix:
        num_repetitive = int(num_samples * repetitive_mix_ratio)
        is_repetitive_list = [True] * num_repetitive + [False] * (num_samples - num_repetitive)
        rng.shuffle(is_repetitive_list)
        print(f"Repetitive mix: {num_repetitive} repetitive, {num_samples - num_repetitive} normal")
    else:
        is_repetitive_list = [repetitive] * num_samples
        print(f"Repetitive: {repetitive}")

    for sample_idx in tqdm(range(num_samples), desc="Creating permutation dataset"):
        for _ in range(10):
            is_rep = is_repetitive_list[sample_idx]
            
            while True:
                state = rng.choices(OBJECT_POOL, k=num_objects)
                if len(set(state)) > 1:
                    break
            
            init_desc = "init " + " ".join(f"{char(i+1)} {obj}" for i, obj in enumerate(state)) + " "
            length = rng.randint(min_length, max_length)
            if include_indices:
                chain_ids = rng.sample(range(1, max_index), length)
                get_curr_id = lambda step: get_idx_str(chain_ids[step])

            ops, trace_steps = [], []
            prev_i = prev_j = None
            init_state = list(state)
            history = {char(i+1): [] for i in range(num_objects)}

            for step in range(length):
                if is_rep and step > 0 and rng.random() < REPETITION_PROBABILITY:
                    i, j = prev_i, prev_j
                else:
                    i, j = rng.sample(range(num_objects), 2)

                val_i, val_j = state[i], state[j]
                
                box1, box2 = i + 1, j + 1
                if delta_cot and val_i != val_j:
                    history[char(box1)].append((val_i, val_j))
                    history[char(box2)].append((val_j, val_i))
                    
                state[i], state[j] = state[j], state[i]
                if include_indices:
                    curr_id = get_curr_id(step)
                else:
                    curr_id = ""

                ops.append(build_op_string(box1, box2, curr_id, include_indices))
                trace_steps.append(build_trace_step(
                    box1, box2, val_i, val_j, state, curr_id, delta_cot, include_indices
                ))
                prev_i, prev_j = i, j

            trace_str = "".join(trace_steps)
            trace_str += "load end "

            if delta_cot:
                res_parts = ["res"]
                for idx in range(num_objects):
                    var_name = char(idx+1)
                    initial_val = init_state[idx]
                    transitions = history[var_name]
                    
                    var_res = f"<{var_name}> init {initial_val}"
                    
                    if not transitions:
                        var_res += f" IN == OUT final {initial_val}"
                    else:
                        # Evaluate ONLY the initial_val
                        in_count = sum(1 for (old, new) in transitions if new == initial_val)
                        out_count = sum(1 for (old, new) in transitions if old == initial_val)
                        
                        if in_count == out_count:
                            relation = "=="
                            final_val = initial_val
                        else:
                            # It left the initial state. 
                            relation = "<"
                            final_val = transitions[-1][1] 
                            
                        # Notice we removed {sym} because it's implicitly always the initial_val now
                        var_res += f" IN {relation} OUT final {final_val}"

                    res_parts.append(var_res)
                trace_str += " ".join(res_parts) + " "

            answer_str = " ".join(state)
            ops_block = " ".join(ops) + " end ."
            prompt_str = f"{init_desc}operation {ops_block}"

            full_text = f"{prompt_str} {TRACE_TOKEN} {trace_str}{FINAL_ANSWER_TOKEN} {answer_str}{END_OF_TEXT_TOKEN}"
            full_text += ""
            h = int.from_bytes(hashlib.blake2b(full_text.encode(), digest_size=16).digest(), "little")
            
            if h not in seen:
                seen.add(h)
                break

        yield {"text": full_text}

def generate_permutation_binary_dataset(
    num_samples: int,
    min_length: int,
    max_length: int,
    seed: int,
    num_objects: int = 5,
    include_indices: bool = True,
    delta_cot: bool = False,
    save_to_path: str = None,
    repetitive: bool = False,
    repetitive_mix: bool = False,
    repetitive_mix_ratio: float = 0.2,
    max_index: int = 70,
):
    if repetitive_mix_ratio > 1.0 or repetitive_mix_ratio < 0.0:
        raise ValueError(
            f"repetitive_mix_ratio must be between 0 and 1. Was: {repetitive_mix_ratio}"
        )
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
            "delta_cot": delta_cot,
            "repetitive_mix": repetitive_mix,
            "repetitive_mix_ratio": repetitive_mix_ratio,
            "repetitive": repetitive,
            "max_index": max_index,
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

if __name__ == "__main__":
    ds_exp = generate_permutation_binary_dataset(
        num_samples=3, min_length=3, max_length=3, seed=44, num_objects=5, random_start_index=True, include_indices=True, delta_cot=True, repetitive=False
    )
    for ex in ds_exp:
        print(ex["text"])
        print("\n")
