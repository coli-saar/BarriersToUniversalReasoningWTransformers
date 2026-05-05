import importlib.util
import os
import random

from lengthgen.constants import TRACE_TOKEN, FINAL_ANSWER_TOKEN, END_OF_TEXT_TOKEN
from datasets import Dataset
from pathlib import Path
from tqdm import tqdm


_spec = importlib.util.spec_from_file_location(
    "sample_formula", Path(__file__).parent / "sample-formula.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_formula_counts = _mod.compute_formula_counts
sample_formula = _mod.sample_formula

def get_idx_str(i: int):
    return f"<{i}>"

def parse_formula(formula_str):
    """Parse a formula string into an AST: atom | ('¬', sub) | (op, left, right)."""
    formula = formula_str.replace(' ', '')

    def at(pos):
        if pos >= len(formula):
            return None, pos
        c = formula[pos]
        if c == '¬':
            sub, p = at(pos + 1)
            return ('¬', sub), p
        if formula[pos:pos + 4] == 'true':
            return 'true', pos + 4
        if formula[pos:pos + 5] == 'false':
            return 'false', pos + 5
        if c == '(':
            left, p  = at(pos + 1)
            op       = formula[p]
            right, p = at(p + 1)
            return (op, left, right), p + 1   # skip ')'
        raise ValueError(f"Unexpected '{c}' at pos {pos}")

    ast, p = at(0)
    if p < len(formula):
        raise ValueError(f"Trailing characters after pos {p}")
    return ast


def formula_to_spaced(formula_str):
    """Insert whitespace between every token in a plain formula string."""
    # Add spaces around operators and parentheses
    spaced = formula_str
    for ch in ['(', ')', '¬', '∧', '∨']:
        spaced = spaced.replace(ch, f' {ch} ')
    # Normalize multiple spaces
    return ' '.join(spaced.split())


def add_post_order_indices(formula_str, starting_index=0, add_child_indices=False):
    ast = parse_formula(formula_str)
    counter = starting_index

    def walk(node):
        nonlocal counter
        if isinstance(node, str):
            idx = counter; counter += 1
            return f"{get_idx_str(idx)} {node}", idx
        if len(node) == 2:
            op, sub = node
            sub_s, sub_i = walk(sub)
            idx = counter; counter += 1
            if add_child_indices:
                return f"[ {get_idx_str(sub_i)} ] {get_idx_str(idx)} {op} {sub_s}", idx
            return f"{get_idx_str(idx)} {op} {sub_s}", idx
        op, left, right = node
        l_s, l_i = walk(left)
        r_s, r_i = walk(right)
        idx = counter; counter += 1
        if add_child_indices:
            return f"( {l_s} [ {get_idx_str(l_i)} , {get_idx_str(r_i)} ] {get_idx_str(idx)} {op} {r_s} )", idx
        return f"( {l_s} {get_idx_str(idx)} {op} {r_s} )", idx

    s, _ = walk(ast)
    return s


def evaluate_with_cot(ast, starting_index=0, include_indices=False):
    cot_steps = []
    counter = starting_index

    def ev(node):
        nonlocal counter
        if isinstance(node, str):
            val = node == 'true'
            tag = f"{get_idx_str(counter)} {'T' if val else 'F'}" if include_indices else ('T' if val else 'F')
            cot_steps.append(tag); counter += 1
            return val
        if len(node) == 2:
            _, sub = node
            res = not ev(sub)
            tag = f"{get_idx_str(counter)} {'T' if res else 'F'}" if include_indices else ('T' if res else 'F')
            cot_steps.append(tag); counter += 1
            return res
        op, left, right = node
        lv, rv = ev(left), ev(right)
        res = (lv and rv) if op == '∧' else (lv or rv)
        tag = f"{get_idx_str(counter)} {'T' if res else 'F'}" if include_indices else ('T' if res else 'F')
        cot_steps.append(tag); counter += 1
        return res

    result, _ = ev(ast), None
    return result, cot_steps


def _boolean_generator(
    num_samples: int,
    min_size: int,
    max_size: int,
    seed: int,
    include_indices: bool,
    random_start_index: bool,
    max_starting_index: int,
):
    rng = random.Random(seed)
    counts = compute_formula_counts(2 * max_size)

    for _ in tqdm(range(num_samples), desc="Creating boolean dataset"):
        size = rng.randint(min_size, max_size)
        f_str = sample_formula(size, counts, rng)

        try:
            ast = parse_formula(f_str)
        except ValueError:
            continue
        
        # for eval we always start indices from 0
        starting_index = (
            rng.randint(0, max_starting_index)
            if (include_indices and random_start_index)
            else 0
        )

        # Build prompt: formula with optional index hints
        if include_indices:
            prompt_formula = add_post_order_indices(
                f_str,
                starting_index=starting_index,
                add_child_indices=True,   # tied to include_indices per spec
            )
        else:
            prompt_formula = formula_to_spaced(f_str)

        # Build trace: post-order CoT steps
        result, cot_steps = evaluate_with_cot(
            ast,
            starting_index=starting_index,
            include_indices=include_indices,
        )

        trace_str  = " ".join(cot_steps)
        answer_str = "T" if result else "F"

        full_text = (
            f"{prompt_formula} {TRACE_TOKEN} {trace_str} "
            f"{FINAL_ANSWER_TOKEN} {answer_str}{END_OF_TEXT_TOKEN}"
        )

        yield {"text": full_text}

def generate_boolean_dataset(
    num_samples: int,
    min_length: int, # maps to min AST size
    max_length: int, # maps to max AST size
    seed: int,
    include_indices: bool = False,
    random_start_index: bool = True,
    max_starting_index: int = 60,
    save_to_path: str = None,
):
    dataset = Dataset.from_generator(
        _boolean_generator,
        gen_kwargs={
            "num_samples": num_samples,
            "min_size": min_length,
            "max_size": max_length,
            "seed": seed,
            "random_start_index": random_start_index,
            "max_starting_index": max_starting_index,
            "include_indices": include_indices,
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
    ds = generate_boolean_dataset(
        num_samples=3, min_length=4, max_length=7, seed=45, 
        random_start_index=False, 
        include_indices=False,
    )
    for ex in ds:
        print(ex["text"])

    ds = generate_boolean_dataset(
        num_samples=3, min_length=4, max_length=7, seed=45, 
        random_start_index=False, 
        include_indices=True,
    )
    for ex in ds:
        print(ex["text"])
