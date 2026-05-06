
import os
import wandb
import re
import shutil
import torch

from lengthgen.constants import TRACE_TOKEN, FINAL_ANSWER_TOKEN, PADDING_TOKEN, END_OF_TEXT_TOKEN
from safetensors.torch import load_file
from lengthgen.tasks import registry as task_registry
from tqdm import tqdm
from transformers import TrainerCallback

def normalize(s):
    if not isinstance(s, str):
        return ""
    s = s.replace(PADDING_TOKEN, "").replace(END_OF_TEXT_TOKEN, "")
    s = re.sub(r'<[^>]+>', '', s)
    s = s.replace('$', '')
    return s.replace(" ", "")

# callback to calculate ood/id val performance
class BatchedAccuracyCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, id_eval_dataset=None, task="permutation",
                 batch_size=32, max_eval_samples=50):
        self.tokenizer = tokenizer
        self.task = task
        self.eval_dataset = eval_dataset
        self.id_eval_dataset = id_eval_dataset
        self.batch_size = batch_size
        self.max_eval_samples = max_eval_samples
        self.eos_id = tokenizer.convert_tokens_to_ids(END_OF_TEXT_TOKEN)

    def extract_answer(self, text):
        # Match 'answer', optional spaces, then capture everything until '<|' or end of string
        match = re.search(rf'{FINAL_ANSWER_TOKEN}\s*(.*?)(?=<\||$)', text)
        if match:
            return match.group(1).strip()
        return None

    def extract_generated_answer(self, generated_text):
        match = re.search(rf'{FINAL_ANSWER_TOKEN}\s*(.*?)(?=<\||$)', generated_text)
        if match:
            return match.group(1).strip()
        return None

    def _evaluate_split(self, model, dataset, split_name):
        num_samples = min(len(dataset), self.max_eval_samples)
        dataset     = dataset.select(range(num_samples))

        correct = 0
        total   = 0

        with torch.no_grad():
            for i in tqdm(range(0, num_samples, self.batch_size), desc=f"[{split_name}] Evaluating"):
                batch_examples = dataset[i : i + self.batch_size]
                full_texts     = batch_examples["text"]

                prompts, ground_truths = [], []
                for full_text in full_texts:
                    ground_truth = self.extract_answer(full_text)
                    if ground_truth is None:
                        continue
                    prompt = full_text.split(TRACE_TOKEN)[0] + TRACE_TOKEN
                    prompts.append(prompt)
                    ground_truths.append(ground_truth)

                if not prompts:
                    continue

                inputs  = self.tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=3096
                ).to("cuda")
                if i == 0:  # Only print for the first batch of the split
                    print(f"[{split_name}] SANITY CHECK: FIRST PROMPT TENSOR (DECODED)")
                    # Grab the first sequence in the batch, remove padding for clean viewing
                    sample_ids = inputs["input_ids"][0]
                    sample_ids = sample_ids[sample_ids != self.tokenizer.pad_token_id]
                    raw_decoded = self.tokenizer.decode(sample_ids, skip_special_tokens=False)
                    print(repr(raw_decoded))
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=3096,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.eos_id,
                )

                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

                for j, text in enumerate(decoded):
                    pred = self.extract_generated_answer(text)

                    if pred == ground_truths[j]:
                        correct += 1
                    if total < 2:
                        print(f"[{split_name}] GT: {ground_truths[j]} | Pred: {text}")
                    total += 1

        accuracy = correct / total if total > 0 else 0.0
        print(f"[{split_name}] Accuracy: {accuracy:.2%} ({correct}/{total})")
        return accuracy, correct, total

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        self.tokenizer.padding_side = "left"
        model.eval()

        # OOD evaluation
        ood_accuracy, ood_correct, ood_total = self._evaluate_split(
            model, self.eval_dataset, "OOD"
        )

        # ID evaluation
        id_accuracy = None
        if self.id_eval_dataset is not None:
            id_accuracy, id_correct, id_total = self._evaluate_split(
                model, self.id_eval_dataset, "ID"
            )

        log_dict = {
            "eval/ood_accuracy": ood_accuracy,
            "eval/ood_correct": ood_correct,
            "step": state.global_step,
        }
        if id_accuracy is not None:
            log_dict["eval/id_accuracy"] = id_accuracy
            log_dict["eval/id_correct"] = id_correct

        wandb.log(log_dict)
        metrics["eval_ood_accuracy"] = ood_accuracy

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        model.train()
        self.tokenizer.padding_side = "right"

# callback to abort runs early when they are above a certain val accuracy threshold
class EarlyAbortCallback(TrainerCallback):
    def __init__(self, abort_step, max_eval_loss):
        self.abort_step = abort_step
        self.max_eval_loss = max_eval_loss

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # We only check if we have reached or passed the target step
        if self.abort_step is not None and state.global_step >= self.abort_step:
            eval_loss = metrics.get("eval_loss")
            
            if eval_loss is not None and eval_loss > self.max_eval_loss:
                print(f"\n[EarlyAbortCallback] Step {state.global_step}: eval_loss {eval_loss:.4f} > {self.max_eval_loss}. Aborting training early!")
                control.should_training_stop = True

# we use a callback to setup the wandb config, as wandb.init/wandb.config seems to lead to issues when training on multiple GPUs
class WandbConfigCallback(TrainerCallback):
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            wandb.config.update(self.config_dict, allow_val_change=True)

# callback to save the best performing checkpoint based on ood val performance
class BestOODCheckpointCallback(TrainerCallback):
    def __init__(self, metric_name = "eval_ood_accuracy", greater_is_better = True):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = float("-inf") if greater_is_better else float("inf")
        self.best_checkpoint_dir = None

    def _is_better(self, current):
        if self.greater_is_better:
            return current > self.best_metric
        return current < self.best_metric
    
    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            for entry in os.scandir(args.output_dir):
                if entry.is_dir() and entry.name.startswith("checkpoint-"):
                    shutil.rmtree(entry.path)
                    print(f"Cleaned up stale checkpoint {entry.path}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        should_save = False
        if state.is_world_process_zero:
            if metrics is not None:
                current = metrics.get(self.metric_name)
                if current is not None and self._is_better(current):
                    old_best_dir = self.best_checkpoint_dir
                    self.best_metric = current
                    self.best_checkpoint_dir = os.path.join(
                        args.output_dir, f"checkpoint-{state.global_step}"
                    )
                    if old_best_dir and os.path.exists(old_best_dir):
                        shutil.rmtree(old_best_dir, ignore_errors=True)
                        print(f"Removed previous best checkpoint: {old_best_dir}")
                    should_save = True
                    print(f"New best {self.metric_name}: {current:.4f} — saving model.")
                else:
                    print(f"No improvement ({self.metric_name}={current:.4f}, best={self.best_metric:.4f}) — skipping save.")

        if torch.distributed.is_initialized():
            flag = torch.tensor(int(should_save), device=args.device)
            torch.distributed.broadcast(flag, src=0)
            should_save = bool(flag.item())

        control.should_save = should_save

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if not state.is_world_process_zero:
            return
            
        if self.best_checkpoint_dir and os.path.exists(self.best_checkpoint_dir):
            safetensors_path = os.path.join(self.best_checkpoint_dir, "model.safetensors")
            pytorch_path = os.path.join(self.best_checkpoint_dir, "pytorch_model.bin")

            if os.path.exists(safetensors_path):
                print(f"Restoring best model from {safetensors_path}")
                state_dict = load_file(safetensors_path, device="cpu")
            elif os.path.exists(pytorch_path):
                print(f"Restoring best model from {pytorch_path}")
                state_dict = torch.load(pytorch_path, map_location="cpu")
            else:
                print("No model weights found in best checkpoint dir, skipping restore.")
                return

            # lm_head.weight is tied to transformer.wte.weight in GPT2 and
            # deduped by safetensors — re-add it before loading
            if "lm_head.weight" not in state_dict and "transformer.wte.weight" in state_dict:
                state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

            model.load_state_dict(state_dict)
            print(f"Restored best model with {self.metric_name}={self.best_metric:.4f}")

            model.save_pretrained(args.output_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(args.output_dir)
            print(f"Saved best model to {args.output_dir}")