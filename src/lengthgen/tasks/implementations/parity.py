import os
import random

from lengthgen.constants import TRACE_TOKEN, FINAL_ANSWER_TOKEN, END_OF_TEXT_TOKEN
from datasets import Dataset
from tqdm import tqdm

def compute_cot_explicit(bits):
    state = 0
    states = []
    for bit in bits:
        state ^= bit
        states.append('E' if state == 0 else 'O')
    return states

def compute_cot_value_change(bits):
    states = ['E'] # initial even parity
    parity_is_odd = False
    for bit in bits:
        if bit == 1:
            parity_is_odd = not parity_is_odd
            states.append('O' if parity_is_odd else 'E')
    return states

def _parity_generator(
    num_samples,
    min_length,
    max_length,
    seed,
    delta_cot,
):
    rng = random.Random(seed)
    cot_fn = compute_cot_value_change if delta_cot else compute_cot_explicit

    for _ in tqdm(range(num_samples), desc="Creating parity dataset"):
        length = rng.randint(min_length, max_length)
        bits   = [rng.randint(0, 1) for _ in range(length)]

        # Prompt: space-separated bit string
        prompt_str = " ".join(str(b) for b in bits)

        # CoT trace
        cot_states = cot_fn(bits)
        trace_str = " ".join(cot_states)

        # Final answer: last CoT state = overall parity
        answer_str = cot_states[-1]

        full_text = (
            f"{prompt_str} {TRACE_TOKEN} {trace_str} "
            f"{FINAL_ANSWER_TOKEN} {answer_str}{END_OF_TEXT_TOKEN}"
        )
        yield {"text": full_text}


def generate_parity_dataset(
    num_samples,
    min_length,
    max_length,
    seed,
    delta_cot = False, # True -> value-change, False -> naive
    save_to_path = None,
):
    dataset = Dataset.from_generator(
        _parity_generator,
        gen_kwargs={
            "num_samples": num_samples,
            "min_length": min_length,
            "max_length": max_length,
            "seed": seed,
            "delta_cot": delta_cot,
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
    ds_exp = generate_parity_dataset(
        num_samples=3, min_length=4, max_length=7, seed=44, delta_cot=False
    )
    for ex in ds_exp:
        print(ex["text"])
        print("\n")
    
    ds_delta = generate_parity_dataset(
        num_samples=3, min_length=4, max_length=7, seed=44, delta_cot=True
    )
    for ex in ds_delta:
        print(ex["text"])
        print("\n")