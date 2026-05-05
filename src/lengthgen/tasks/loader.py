# tasks/loader.py
import inspect
from lengthgen.tasks.registry import TaskType, TASK_CONFIGS

def _call_generator(generator_fn, **all_params):
    sig = inspect.signature(generator_fn)
    # check for **kwargs keyword in signature
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if has_var_keyword:
        return generator_fn(**all_params)
    
    accepted = set(sig.parameters.keys())
    unused = set(all_params) - accepted
    if unused:
        print(f"{generator_fn.__name__} ignoring unused params: {unused}")
    
    return generator_fn(**{k: v for k, v in all_params.items() if k in accepted})

def load_split(
    task: TaskType,
    split: str, # "train" | "val"
    num_samples: int,
    min_length: int,
    max_length: int,
    seed: int,
    save_to_path: str = None,
    **runtime_overrides, # e.g. from a sweep
):
    cfg = TASK_CONFIGS[task]
    
    params = cfg.base_params.copy()
    params.update(cfg.train_params if split == "train" else cfg.val_params)
    params.update(runtime_overrides) # sweep-level overrides win
    
    return _call_generator(
        cfg.generator,
        num_samples=num_samples,
        min_length=min_length,
        max_length=max_length,
        seed=seed,
        save_to_path=save_to_path,
        **params,
    )