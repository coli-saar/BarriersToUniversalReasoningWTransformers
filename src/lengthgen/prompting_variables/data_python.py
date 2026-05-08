import json
import random
import argparse

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union


# =========================
# Core data structures
# =========================

@dataclass
class Operation:
    target: str
    op_type: str   # Only "assign_const" is used
    value: int


@dataclass
class ProgramExample:
    variables: List[str]             # The target variables being queried
    distractor_vars: List[str]       # The irrelevant variables 
    init_state: Dict[str, int]
    operations: List[Operation]
    final_state: Dict[str, int]
    query_order: List[str]
    program_lines: List[str]


# =========================
# File helpers
# =========================

def save_records_jsonl(path: Union[str, Path], rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_jsonl(path: Union[str, Path], examples: List[ProgramExample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(example_to_json_record(ex), ensure_ascii=False) + "\n")


# =========================
# Dataset generator
# =========================

class VariableTrackingDatasetGenerator:
    def __init__(
        self,
        seed: int = 0,
        var_prefix: str = "node",
        const_values: Tuple[int, ...] = tuple(range(10)), # Default to single digit numbers (0-9)
    ):
        self.rng = random.Random(seed)
        self.var_prefix = var_prefix
        self.const_values = const_values

    def make_variables(self, n_vars: int) -> List[str]:
        letters = "abcdefghijklmnopqrstuvwxyz"
        if n_vars <= 26:
            return [f"{self.var_prefix}_{letters[i]}" for i in range(n_vars)]
        return [f"{self.var_prefix}_{i}" for i in range(n_vars)]

    def sample_initial_state(self, variables: List[str]) -> Dict[str, int]:
        return {v: self.rng.choice(self.const_values) for v in variables}

    def sample_assign_operation(
        self,
        variables: List[str],
        force_change_prob: float = 0.0,
        current_state: Optional[Dict[str, int]] = None,
    ) -> Operation:
        if current_state is None:
            current_state = {}

        target = self.rng.choice(variables)
        
        if force_change_prob > 0 and self.rng.random() < force_change_prob and target in current_state:
            changing_values = [v for v in self.const_values if v != current_state[target]]
            if changing_values:
                return Operation(target=target, op_type="assign_const", value=self.rng.choice(changing_values))

        return Operation(target=target, op_type="assign_const", value=self.rng.choice(self.const_values))

    def apply_operation(self, state: Dict[str, int], op: Operation) -> Tuple[int, int]:
        old_value = state[op.target]
        new_value = op.value
        state[op.target] = new_value
        return old_value, new_value

    def operation_to_code(self, op: Operation) -> str:
        return f"{op.target} = {op.value}"

    def generate_example(
        self,
        n_vars: int = 3,
        n_ops: int = 6,
        n_distractors: int = 10,
        mode: str = "spread",
        shuffle_query_order: bool = False,
        force_change_prob: float = 0.0,
    ) -> ProgramExample:
        # Generate both target variables and distractor variables, then randomly shuffle them
        all_vars = self.make_variables(n_vars + n_distractors)
        self.rng.shuffle(all_vars)
        
        variables = sorted(all_vars[:n_vars])
        distractor_vars = sorted(all_vars[n_vars:])
        
        init_state = self.sample_initial_state(all_vars)
        
        # 1. Generate target operations
        target_ops = []
        sim_state = dict(init_state)
        for _ in range(n_ops):
            op = self.sample_assign_operation(variables, force_change_prob, sim_state)
            target_ops.append(op)
            self.apply_operation(sim_state, op)
            
        # 2. Generate distractor operations
        distractor_ops = []
        d_state = dict(init_state)
        for _ in range(n_distractors):
            op = self.sample_assign_operation(distractor_vars, force_change_prob, d_state)
            distractor_ops.append(op)
            self.apply_operation(d_state, op)
            
        # 3. Mix the operations according to the mode
        mixed_ops = []
        if mode == "spread":
            t_len, d_len = len(target_ops), len(distractor_ops)
            total = t_len + d_len
            indices = set(int(i * total / t_len) for i in range(t_len)) if t_len > 0 else set()
            
            t_idx, d_idx = 0, 0
            for i in range(total):
                if i in indices and t_idx < t_len:
                    mixed_ops.append(target_ops[t_idx])
                    t_idx += 1
                else:
                    mixed_ops.append(distractor_ops[d_idx])
                    d_idx += 1
        else: # clustered
            insert_idx = self.rng.randint(0, len(distractor_ops))
            mixed_ops = distractor_ops[:insert_idx] + target_ops + distractor_ops[insert_idx:]

        # 4. Finalize the state map
        state = dict(init_state)
        operations: List[Operation] = []
        program_lines: List[str] = []

        # Write initialization
        for v, val in init_state.items():
            program_lines.append(f"{v} = {val}")

        # Write mixed operations
        for op in mixed_ops:
            self.apply_operation(state, op)
            operations.append(op)
            program_lines.append(self.operation_to_code(op))

        final_state = dict(state)
        query_order = list(variables)
        if shuffle_query_order:
            self.rng.shuffle(query_order)

        return ProgramExample(
            variables=variables,
            distractor_vars=distractor_vars,
            init_state=init_state,
            operations=operations,
            final_state=final_state,
            query_order=query_order,
            program_lines=program_lines,
        )

    def generate_dataset(
        self,
        n_examples: int,
        n_vars: int = 3,
        n_ops: int = 6,
        n_distractors: int = 15,
        shuffle_query_order: bool = False,
        force_change_prob: float = 0.0,
    ) -> List[ProgramExample]:
        
        examples = []
        # 50% Spread out, 50% Clustered together
        half = n_examples // 2
        for i in range(n_examples):
            mode = "spread" if i < half else "clustered"
            examples.append(
                self.generate_example(
                    n_vars=n_vars,
                    n_ops=n_ops,
                    n_distractors=n_distractors,
                    mode=mode,
                    shuffle_query_order=shuffle_query_order,
                    force_change_prob=force_change_prob,
                )
            )
        return examples


# =========================
# Serialization helpers
# =========================

def example_to_json_record(ex: ProgramExample) -> Dict[str, Any]:
    return {
        "variables": ex.variables,
        "distractor_vars": ex.distractor_vars,
        "init_state": ex.init_state,
        "operations": [asdict(op) for op in ex.operations],
        "final_state": ex.final_state,
        "query_order": ex.query_order,
        "program_lines": ex.program_lines,
    }


# =========================
# Prompt rendering
# =========================

@dataclass
class RenderConfig:
    include_linenums: bool = False
    include_value_change: bool = False
    output_tag_name: str = "output"
    use_print_query: bool = True


class PromptRenderer:
    @staticmethod
    def add_linenums(lines: List[str]) -> List[str]:
        width = len(str(len(lines)))
        return [f"{i:>{width}}. {line}" for i, line in enumerate(lines, start=1)]

    @staticmethod
    def render_query(ex: ProgramExample, use_print_query: bool = True) -> str:
        if use_print_query:
            return "print(" + ", ".join(v for v in ex.query_order) + ")"
        return "What are the final values of: " + ", ".join(ex.query_order) + "?"

    @staticmethod
    def render_gold_answer(ex: ProgramExample, tag_name: str = "output") -> str:
        body = " ".join(str(ex.final_state[v]) for v in ex.query_order)
        return f"<{tag_name}> {body} </{tag_name}>"

    @staticmethod
    def render_program_with_optional_inline_value_change(
        ex: ProgramExample,
        include_linenums: bool = False,
        include_value_change: bool = False,
    ) -> str:
        rendered_lines: List[str] = []
        state = dict(ex.init_state)
        init_len = len(ex.init_state)

        for line in ex.program_lines[:init_len]:
            rendered_lines.append(line)

        for step_idx, op in enumerate(ex.operations, start=1):
            code_line = ex.program_lines[init_len + step_idx - 1]

            if include_value_change:
                old_value = state[op.target]
                new_value = op.value
                state[op.target] = new_value

                if old_value == new_value:
                    rendered_lines.append(f"{code_line}    # {op.target}: no change")
                else:
                    rendered_lines.append(f"{code_line}    # {op.target}: {old_value} -> {new_value}")
            else:
                state[op.target] = op.value
                rendered_lines.append(code_line)

        if include_linenums:
            rendered_lines = PromptRenderer.add_linenums(rendered_lines)

        return "\n".join(rendered_lines)

    def render_direct_prompt(
        self,
        ex: ProgramExample,
        config: RenderConfig,
    ) -> str:
        parts: List[str] = []
        parts.append("Program:")
        parts.append(
            self.render_program_with_optional_inline_value_change(
                ex,
                include_linenums=config.include_linenums,
                include_value_change=config.include_value_change,
            )
        )
        parts.append("")
        parts.append("Query:")
        parts.append(self.render_query(ex, use_print_query=config.use_print_query))
        parts.append(
            f"Return only the final answer in <{config.output_tag_name}> </{config.output_tag_name}> tags."
        )
        return "\n".join(parts)


# =========================
# Rendered record helpers
# =========================

def variant_suffix(include_linenums: bool, include_value_change: bool) -> str:
    if include_linenums and include_value_change:
        return "linenums+value_change"
    if include_linenums:
        return "linenums"
    return "none"


def make_rendered_record(
    example_id: int,
    ex: ProgramExample,
    renderer: PromptRenderer,
    config: RenderConfig,
) -> Dict[str, Any]:
    suffix = variant_suffix(
        include_linenums=config.include_linenums,
        include_value_change=config.include_value_change,
    )

    return {
        "example_id": example_id,
        "condition": f"direct_{suffix}",
        "variant_suffix": suffix,
        "variables": ex.variables,
        "query_order": ex.query_order,
        "final_state": ex.final_state,
        "program_lines": ex.program_lines,
        "prompt": renderer.render_direct_prompt(ex, config),
        "gold_answer": renderer.render_gold_answer(ex, tag_name=config.output_tag_name),
        "output_tag_name": config.output_tag_name,
        "include_linenums": config.include_linenums,
        "include_value_change": config.include_value_change,
    }


# =========================
# Bundle export
# =========================

def export_direct_answer_bundle(
    dataset: List[ProgramExample],
    args: argparse.Namespace,
) -> None:
    base_dir = Path(args.output_root) / f"len_{args.n_ops}"

    # Save the latent dataset
    latent_path = base_dir / "latent" / "dataset.jsonl"
    save_jsonl(latent_path, dataset)

    renderer = PromptRenderer()
    
    # We now only want 3 specific variants:
    # 1. none (False, False)
    # 2. linenums (True, False)
    # 3. linenums+value_change (True, True)
    variants = [
        (False, False),  
        (True, False),   
        (True, True),    
    ]

    # Export rendered direct evaluation formats
    direct_dir = base_dir / "final" / "direct"
    for include_linenums, include_value_change in variants:
        suffix = variant_suffix(include_linenums, include_value_change)
        config = RenderConfig(
            include_linenums=include_linenums,
            include_value_change=include_value_change,
            output_tag_name=args.output_tag_name,
            use_print_query=args.use_print_query,
        )
        rows = [
            make_rendered_record(
                example_id=i,
                ex=ex,
                renderer=renderer,
                config=config,
            )
            for i, ex in enumerate(dataset)
        ]
        save_records_jsonl(direct_dir / f"{suffix}.jsonl", rows)

    # Preview generation
    if args.preview > 0:
        print("=" * 88)
        print("direct")
        for suffix in ["none", "linenums", "linenums+value_change"]:
            path = direct_dir / f"{suffix}.jsonl"
            rows = []
            with open(path, encoding="utf-8") as f:
                for _, line in zip(range(args.preview), f):
                    rows.append(json.loads(line))
            print("-" * 88)
            print(path)
            for row in rows:
                print(f"example_id={row['example_id']} condition={row['condition']}")
                print(row["prompt"])
                print("GOLD:", row["gold_answer"])
                print()


# =========================
# CLI
# =========================

def str2bool(v: str) -> bool:
    v = v.lower()
    if v in {"yes", "true", "t", "1", "y"}:
        return True
    if v in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate variable-tracking datasets for the direct-answer setup."
    )

    parser.add_argument("--output-root", type=str, default="datasets")

    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--n-vars", type=int, default=3)
    parser.add_argument("--n-ops", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--var-prefix", type=str, default="node")

    parser.add_argument("--shuffle-query-order", type=str2bool, default=False)
    parser.add_argument("--force-change-prob", type=float, default=0.8)

    parser.add_argument("--use-print-query", type=str2bool, default=True)
    parser.add_argument("--output-tag-name", type=str, default="output")

    parser.add_argument("--preview", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gen = VariableTrackingDatasetGenerator(
        seed=args.seed,
        var_prefix=args.var_prefix,
    )

    dataset = gen.generate_dataset(
        n_examples=args.n_examples,
        n_vars=args.n_vars,
        n_ops=args.n_ops,
        n_distractors=args.n_ops, # equal number of distractors. 
        shuffle_query_order=args.shuffle_query_order,
        force_change_prob=args.force_change_prob,
    )

    export_direct_answer_bundle(dataset, args)


if __name__ == "__main__":
    main()