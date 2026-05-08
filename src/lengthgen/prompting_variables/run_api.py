import argparse
import json
import os
import tqdm
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from together import Together


# ============================================================
# Model helpers
# ============================================================

def get_model_str(model: str) -> str:
    all_models = {
        "llama70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "qwen7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "mistral24B": "okraus2_80f7/mistralai/Mistral-Small-24B-Instruct-2501-bc7fb23e",
    }
    if model not in all_models:
        raise argparse.ArgumentTypeError(
            "Invalid model. Options: llama70B, llama8B, qwen7B, qwen70B, mistral, mistral24B"
        )
    return all_models[model]


def sanitize_model_name(model_alias: str) -> str:
    return model_alias


# ============================================================
# Utilities
# ============================================================

def str2bool(v: str) -> bool:
    v = v.lower()
    if v in {"yes", "true", "t", "1", "y"}:
        return True
    if v in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# Together API
# ============================================================

def get_client() -> Together:
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")
    return Together(api_key=api_key)


def query_model(
    client: Together,
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    retries: int = 3,
    retry_delay: float = 5.0,
    system_prompt: Optional[str] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            text = ""
            if response.choices and response.choices[0].message:
                text = response.choices[0].message.content or ""

            raw = None
            try:
                raw = response.model_dump()
            except Exception:
                try:
                    raw = dict(response)
                except Exception:
                    raw = None

            return text, raw

        except Exception as e:
            if attempt < retries - 1:
                print(f"API error (attempt {attempt + 1}/{retries}): {e}")
                time.sleep(retry_delay)
            else:
                print(f"API error after {retries} attempts: {e}")
                return "", {"error": str(e)}

    return "", None


# ============================================================
# Answer extraction / evaluation
# ============================================================

def extract_between_tags(text: str, tag_name: str = "output") -> Optional[str]:
    pattern = rf"<{re.escape(tag_name)}>\s*(.*?)\s*</{re.escape(tag_name)}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def parse_value_sequence(answer_str: Optional[str]) -> List[str]:
    """
    For answers like:
      <output> 1 0 1 </output>
    or
      <output>1, 0, 1</output>
    return:
      ["1", "0", "1"]
    """
    if not answer_str:
        return []
    
    # Replace commas with spaces to handle comma-separated values
    cleaned_str = answer_str.replace(",", " ")
    
    # Split by whitespace
    return [tok for tok in cleaned_str.split() if tok]


def exact_match_output(pred_text: Optional[str], gold_text: Optional[str]) -> bool:
    pred = parse_value_sequence(pred_text)
    gold = parse_value_sequence(gold_text)
    return len(gold) > 0 and pred == gold


# ============================================================
# Dataset selection
# ============================================================

def select_subset(
    rows: List[Dict[str, Any]],
    first_n: Optional[int] = None,
    sample_n: Optional[int] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    ids: Optional[List[int]] = None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    selected = list(rows)

    if ids:
        id_set = set(ids)
        selected = [row for i, row in enumerate(selected) if row.get("example_id", i) in id_set]

    if start_idx is not None or end_idx is not None:
        s = 0 if start_idx is None else start_idx
        e = len(selected) if end_idx is None else end_idx
        selected = selected[s:e]

    if first_n is not None:
        selected = selected[:first_n]

    if sample_n is not None:
        rng = random.Random(seed)
        sample_n = min(sample_n, len(selected))
        selected = rng.sample(selected, sample_n)

    return selected


def parse_ids(ids_str: Optional[str]) -> Optional[List[int]]:
    if not ids_str:
        return None
    return [int(x.strip()) for x in ids_str.split(",") if x.strip()]


# ============================================================
# Paths
# ============================================================

def build_input_path(
    dataset_root: Path,
    n_ops: int,
    variant: str,
) -> Path:
    # Removed prompt_family logic, defaulting to 'direct'
    return dataset_root / f"len_{n_ops}" / "final" / "direct" / f"{variant}.jsonl"


def build_output_paths(
    results_root: Path,
    n_ops: int,
    model_alias: str,
    variant: str,
    subset_tag: str,
) -> Tuple[Path, Path]:
    base = results_root / f"len_{n_ops}" / "final" / "direct" / sanitize_model_name(model_alias)
    json_path = base / f"{variant}{subset_tag}.json"
    summary_path = base / f"{variant}{subset_tag}_summary.json"
    return json_path, summary_path


def make_subset_tag(
    *,
    first_n: Optional[int],
    sample_n: Optional[int],
    start_idx: Optional[int],
    end_idx: Optional[int],
    ids: Optional[List[int]],
    seed: int,
) -> str:
    parts: List[str] = []

    if first_n is not None:
        parts.append(f"first{first_n}")
    if sample_n is not None:
        parts.append(f"sample{sample_n}_seed{seed}")
    if start_idx is not None or end_idx is not None:
        s = "none" if start_idx is None else str(start_idx)
        e = "none" if end_idx is None else str(end_idx)
        parts.append(f"slice_{s}_{e}")
    if ids:
        joined = "-".join(str(x) for x in ids)
        parts.append(f"ids_{joined}")

    if not parts:
        return ""
    return "__" + "__".join(parts)


# ============================================================
# Experiment loop
# ============================================================

def run_experiment(
    client: Together,
    rows: List[Dict[str, Any]],
    model: str,
    output_tag_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retries: int,
    retry_delay: float,
    system_prompt: Optional[str],
    debug_samples: int = 3,
    sleep_secs: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    correct = 0

    for idx, row in tqdm.tqdm(enumerate(rows), total=len(rows)):
        prompt = row["prompt"]
        gold_answer = row["gold_answer"]

        raw_text, raw_api_response = query_model(
            client=client,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            retries=retries,
            retry_delay=retry_delay,
            system_prompt=system_prompt,
        )

        filtered = extract_between_tags(raw_text, tag_name=output_tag_name)
        gold_filtered = extract_between_tags(gold_answer, tag_name=output_tag_name)

        pred_values = parse_value_sequence(filtered)
        gold_values = parse_value_sequence(gold_filtered)
        is_correct = exact_match_output(filtered, gold_filtered)

        if is_correct:
            correct += 1

        result = {
            "local_index": idx,
            "example_id": row.get("example_id", idx),
            "condition": row.get("condition"),
            "variant_suffix": row.get("variant_suffix"),
            "model": model,

            "prompt": prompt,

            "gold_answer_raw": gold_answer,
            "gold_answer_filtered": gold_filtered,
            "gold_answer_values": gold_values,

            "raw_response_text": raw_text,
            "filtered_response_text": filtered,
            "filtered_response_values": pred_values,

            "correct": is_correct,
            "raw_api_response": raw_api_response,
        }
        results.append(result)

        if idx < debug_samples:
            print("=" * 80)
            print(f"Example {idx} / example_id={result['example_id']}")
            print("-" * 80)
            print("Condition:", result["condition"])
            print("Gold:", gold_answer)
            print("Pred filtered:", filtered)
            print("Correct:", is_correct)
            print("Raw preview:", raw_text[:400].replace("\n", "\\n"))

        if sleep_secs > 0:
            time.sleep(sleep_secs)

    summary = {
        "model": model,
        "n_examples": len(rows),
        "n_correct": correct,
        "accuracy": (correct / len(rows)) if rows else 0.0,
        "output_tag_name": output_tag_name,
    }
    return results, summary


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Together experiments on rendered variable-tracking datasets."
    )

    # dataset structure
    parser.add_argument("--dataset-root", type=str, default="datasets")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--n-ops", type=int, required=True)
    
    # Updated choices exactly per request
    parser.add_argument("--variant", type=str, required=True, choices=["none", "linenums", "linenums+value_change"])

    # model / api
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a very careful and precise assistant. You always follow the instructions and solve tasks yourself. You never generate code. You also give the answer directly whenever possible without trying to generate any intermediate Chain of Thought.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--sleep-secs", type=float, default=0.0)

    # subset selection
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--ids", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    # output parsing / debug
    parser.add_argument("--output-tag-name", type=str, default="output")
    parser.add_argument("--debug-samples", type=int, default=3)
    parser.add_argument("--save-jsonl", type=str2bool, default=False)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = build_input_path(
        dataset_root=Path(args.dataset_root),
        n_ops=args.n_ops,
        variant=args.variant,
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    rows = load_jsonl(input_path)

    ids = parse_ids(args.ids)
    selected = select_subset(
        rows,
        first_n=args.first_n,
        sample_n=args.sample_n,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        ids=ids,
        seed=args.seed,
    )

    subset_tag = make_subset_tag(
        first_n=args.first_n,
        sample_n=args.sample_n,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        ids=ids,
        seed=args.seed,
    )

    results_json_path, summary_json_path = build_output_paths(
        results_root=Path(args.results_root),
        n_ops=args.n_ops,
        model_alias=args.model,
        variant=args.variant,
        subset_tag=subset_tag,
    )

    results_jsonl_path = results_json_path.with_suffix(".jsonl")

    client = get_client()
    together_model_str = get_model_str(args.model)

    results, summary = run_experiment(
        client=client,
        rows=selected,
        model=together_model_str,
        output_tag_name=args.output_tag_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        retries=args.retries,
        retry_delay=args.retry_delay,
        system_prompt=args.system_prompt,
        debug_samples=args.debug_samples,
        sleep_secs=args.sleep_secs,
    )

    summary.update(
        {
            "model_alias": args.model,
            "together_model": together_model_str,
            "input_path": str(input_path),
            "results_json_path": str(results_json_path),
            "summary_json_path": str(summary_json_path),
            "variant": args.variant,
            "n_ops": args.n_ops,
            "subset_tag": subset_tag,
        }
    )

    save_json(results_json_path, results)
    save_json(summary_json_path, summary)

    if args.save_jsonl:
        save_jsonl(results_jsonl_path, results)

    print("=" * 80)
    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()