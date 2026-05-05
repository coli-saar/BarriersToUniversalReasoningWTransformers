from enum import Enum
from dataclasses import dataclass, field
from typing import Callable

from .implementations.addition import generate_addition_dataset
from .implementations.boolean_eval import generate_boolean_dataset
from .implementations.multiplication import generate_multiplication_dataset
from .implementations.parity import generate_parity_dataset
from .implementations.permutation import generate_permutation_dataset
from .implementations.permutation_new import generate_permutation_new_dataset
from .implementations.permutation_binary import generate_permutation_binary_dataset
from .implementations.permutation_prompting import generate_permutation_prompting_dataset
from .implementations.variable_tracking import generate_variable_tracking_dataset
from .implementations.position_tracking import generate_position_tracking_dataset

class TaskType(str, Enum):
    PERMUTATION = "permutation"
    PERMUTATION_NEW = "permutation_new"
    PERMUTATION_PROMPTING = "permutation_prompting"
    PERMUTATION_BINARY = "permutation_binary"
    ADDITION = "addition"
    ADDITION_NEW = "addition_new"
    MULTIPLICATION = "multiplication"
    BOOLEAN = "boolean"
    PARITY = "parity"
    VARIABLE = "variable_tracking"
    POSITION = "position_tracking"


@dataclass
class TaskConfig:
    generator: Callable
    base_params: dict = field(default_factory=dict)
    
    # Params that vary between train and val
    train_params: dict = field(default_factory=dict)
    val_params: dict = field(default_factory=dict)


TASK_CONFIGS = {
    TaskType.PARITY: TaskConfig(
        generator=generate_parity_dataset,
        base_params=dict(
            delta_cot=False,
        ),
        train_params=dict(),
        val_params=dict()
    ),
    TaskType.BOOLEAN: TaskConfig(
        generator=generate_boolean_dataset,
        base_params=dict(
            include_indices=False
        ),
        train_params=dict(
            random_start_index=True,
        ),
        val_params=dict(
            random_start_index=False,
        )
    ),
    TaskType.PERMUTATION: TaskConfig(
        generator=generate_permutation_dataset,
        base_params=dict(
            num_objects=5,
            include_indices=False,
            delta_cot=False,
        ),
        train_params=dict(
            random_start_index=True,
            repetitive_mix=True,
            repetitive_mix_ratio=0.2,
            shuffle_operations=True,
        ),
        val_params=dict(
            random_start_index=False,
            shuffle_operations=False,
            eval=True,
        ),
    ),
    TaskType.PERMUTATION_BINARY: TaskConfig(
        generator=generate_permutation_binary_dataset,
        base_params=dict(
            num_objects=5,
            include_indices=False,
            delta_cot=False,
        ),
        train_params=dict(
            random_start_index=True,
            repetitive_mix=True,
            repetitive_mix_ratio=0.25,
            shuffle_operations=True,
        ),
        val_params=dict(

        ),
    ),
}

def get_task_generator(task_name):
    if isinstance(task_name, str):
        try:
            task_name = TaskType(task_name)
        except ValueError:
            raise ValueError(f"Task '{task_name}' not found. Available: {[t.value for t in TaskType]}")

    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Generator for {task_name} is not registered.")
        
    return TASK_CONFIGS[task_name].generator