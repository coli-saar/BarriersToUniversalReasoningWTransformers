import hashlib
import os
import random

from lengthgen.constants import TRACE_TOKEN, FINAL_ANSWER_TOKEN, END_OF_TEXT_TOKEN
from datasets import Dataset, disable_caching
from tqdm import tqdm

'''
# 1. Non-signpost (Baseline)
permutation_baseline_kwargs = {
    "include_indices": False,
    "add_tape": False,
    "shuffle_ops": False,
    "add_load": False
}

# 2. Signpost with Tape + Shuffling - Format B in paper
permutation_tape_shuffle_kwargs = {
    "include_indices": True,
    "add_tape": True,
    "shuffle_ops": True, -> at test time set this to False
    "add_load": False
}

# 3. Signpost with Loading logic (No tape/shuffling) - Format A in paper
permutation_load_kwargs = {
    "include_indices": True,
    "add_tape": False,
    "shuffle_ops": False,
    "add_load": True
}

'''

OBJECT_POOL = [
    "Cat", "Dog", "Apple", "Book", "Hat",
    "Monkey", "Dragon", "Spider",
    "Armadillo", "Salamander", "Alligator"
]
INDEX_TO_CHAR = {i: chr(ord("A") + i - 1) for i in range(1, 6)}
# probability for which an operation is repeated (if the sample is selected as repetitive)
REPETITION_PROBABILITY = 0.9

def get_idx_str(i: int):
    return f"<{i}>"

def char(i: int):
    return INDEX_TO_CHAR[i]

def build_op_string(box1: int, box2: int, curr_id: str = "", include_indices: bool = False):
    label = f"{curr_id} " if include_indices else ""
    return f"{label}swap {char(box1)} {char(box2)} {label}."

def build_trace_step(
    box1: int, box2: int, state: list[str], curr_id: str,
    include_indices: bool, add_load: bool
):
    indexed_state = " ".join(f"{char(k+1)} {obj}" for k, obj in enumerate(state))
    label = f"line {curr_id} " if include_indices else ""

    step_str = f"{label}swap {char(box1)} {char(box2)} write {indexed_state} . "
    if add_load:
        return f"load {curr_id} . {step_str}"
    return step_str

def _permutation_generator(
    num_samples: int,
    min_length: int,
    max_length: int,
    seed: int,
    num_objects: int = 5,
    include_indices: bool = False,
    add_tape: bool = False,
    shuffle_ops: bool = False,
    add_load: bool = False,
    repetitive: bool = False,
    repetitive_mix: bool = False,
    repetitive_mix_ratio: float = 0.2,
    max_index: int = 60
):
    print(f"max index: {max_index}")
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
            state = rng.sample(OBJECT_POOL, num_objects)
            init_desc = "init " + " ".join(f"{char(i+1)} {obj}" for i, obj in enumerate(state)) + " "
            length = rng.randint(min_length, max_length)

            # we mix in 25% ordered data during training (if shuffling is requested)
            use_sequential = rng.random() < 0.25
            chain_ids = rng.sample(range(1, max_index), length)
            get_curr_id = lambda step: get_idx_str(chain_ids[step])
            ops, trace_steps = [], []
            prev_i = prev_j = None
            init_state = list(state)
            history = {char(i+1): [] for i in range(num_objects)}

            for step in range(length):
                # Pick swap indices (repeat previous with 90% chance if repetitive)
                if is_rep and step > 0 and rng.random() < REPETITION_PROBABILITY:
                    i, j = prev_i, prev_j
                else:
                    i, j = rng.sample(range(num_objects), 2)

                state[i], state[j] = state[j], state[i]
                box1, box2 = i + 1, j + 1
                curr_id = get_curr_id(step)

                ops.append(build_op_string(box1, box2, curr_id=curr_id, include_indices=include_indices))

                trace_steps.append(build_trace_step(
                    box1, box2, state, curr_id,
                    include_indices=include_indices, add_load=add_load
                ))

                prev_i, prev_j = i, j

            # Assemble prompt and answer
            trace_str = "".join(trace_steps)
            if add_load:
                trace_str += "load end . "
            trace_str += "end "

            answer_str = " ".join(state)
            should_shuffle = shuffle_ops and include_indices and not is_rep and not use_sequential
            if should_shuffle:
                rng.shuffle(ops)

            ops_block = " ".join(ops)
            if add_tape:
                tape_str = "tape " + " ".join(get_idx_str(chain_ids[i]) for i in range(length)) + " end "
                ops_block += " end ."
                prompt_str = f"{init_desc}{tape_str}operation {ops_block}"
            else:
                ops_block += " end ."
                prompt_str = f"{init_desc}operation {ops_block}"

            text = ({"text": f"{prompt_str} {TRACE_TOKEN} {trace_str}{FINAL_ANSWER_TOKEN} {answer_str}{END_OF_TEXT_TOKEN}"})
            h = int.from_bytes(hashlib.blake2b(text["text"].encode(), digest_size=16).digest(), "little")
            if h not in seen:
                seen.add(h)
                break

        yield text

def generate_permutation_dataset(
    num_samples: int,
    min_length: int,
    max_length: int,
    seed: int,
    num_objects: int = 5,
    include_indices: bool = False,
    add_tape: bool = False,
    shuffle_ops: bool = False,
    add_load: bool = False,
    save_to_path: str = None,
    repetitive: bool = False,
    repetitive_mix: bool = False,
    repetitive_mix_ratio: float = 0.2,
    max_index: int = 60,
):
    if repetitive_mix_ratio > 1.0 or repetitive_mix_ratio < 0.0:
        raise ValueError(
            f"repetitive_mix_ratio must be between 0 and 1. Was: {repetitive_mix_ratio}"
        )

    if not include_indices and (add_tape or shuffle_ops or add_load):
        raise ValueError(
            "Invalid configuration: `add_tape`, `shuffle_ops`, and `add_load` "
            "can only be True if `include_indices` is also True."
        )

    if add_load and (add_tape or shuffle_ops):
        raise ValueError(
            "Invalid configuration: The loading logic format (`add_load`) cannot be "
            "combined with the tape and shuffling format (`add_tape` or `shuffle_ops`)."
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
            "add_tape": add_tape,
            "shuffle_ops": shuffle_ops,
            "add_load": add_load,
            "repetitive_mix": repetitive_mix,
            "repetitive_mix_ratio": repetitive_mix_ratio,
            "repetitive": repetitive,
            "max_index": max_index
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
    ds_exp = generate_permutation_dataset(
        num_samples=3, min_length=4, max_length=4, seed=44, num_objects=5, 
        include_indices=True, add_tape=True, shuffle_ops=False, add_load=False
    )
    for ex in ds_exp:
        print(ex["text"])
        print("\n")
