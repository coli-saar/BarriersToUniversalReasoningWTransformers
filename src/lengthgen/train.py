import fire
import json
import os
import random
import torch
import wandb
from pathlib import Path
from datetime import timedelta
from datasets import load_from_disk, disable_caching
from transformers import (
    Trainer, TrainingArguments,
    AutoConfig,
    AutoModelForCausalLM,
    set_seed
)
from lengthgen.tasks.loader import load_split
from trl import DataCollatorForCompletionOnlyLM
from lengthgen.callbacks import BatchedAccuracyCallback, EarlyAbortCallback, WandbConfigCallback, BestOODCheckpointCallback
from lengthgen.collators import RandomOffsetCollator
from lengthgen.util import create_weighted_loss
from lengthgen.build_tokenizer import AlgorithmicTaskTokenizer, TASK_REGISTRY, TASK_TO_TOKENIZER_DIR
from lengthgen.paths import MODEL_CONFIG_BASE_PATH, TOKENIZER_BASE_PATH, DATA_BASE_PATH, MODELS_OUT_BASE_PATH
from lengthgen.constants import TRACE_TOKEN, PositionalEncodings

LOGGING_STEPS = 100
EVAL_STEPS = 800
VAL_SAMPLES = 1000
MAX_EVAL_SAMPLES = 500
EVAL_BATCH_SIZE = 128
DATA_LOADER_NUM_WORKERS = 4

class PosDropoutGPT2(torch.nn.Module):
    def __init__(self, model, tokenizer, pos_dropout_prob=0.1, seed=42):
        super().__init__()
        self.model = model
        self.pos_dropout_prob = pos_dropout_prob
        self._input_ids = None
        self.separator_token_id = tokenizer.convert_tokens_to_ids(TRACE_TOKEN)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.rng = random.Random(seed)
        self._register_hook()

    def _register_hook(self):
        def hook(module, input, output):
            if not self.training:
                return output

            batch_size, seq_len, hidden_size = output.shape
            mask = torch.ones(batch_size, seq_len, 1, device=output.device)

            if self._input_ids is not None:
                if self.rng.random() < self.pos_dropout_prob:
                    for b in range(batch_size):
                        # Find first occurrence of ### token in the sequence
                        sep_positions = (self._input_ids[b] == self.separator_token_id).nonzero(as_tuple=True)[0]
                        prompt_len = sep_positions[0].item() if len(sep_positions) > 0 else seq_len

                        token_mask = (torch.rand(prompt_len, 1, generator=self.generator) 
                                    > self.pos_dropout_prob).float().to(output.device)
                        mask[b, :prompt_len, :] = token_mask

            return output * mask

        self.model.transformer.wpe.register_forward_hook(hook)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        self._input_ids = input_ids
        result = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)
        self._input_ids = None
        return result

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Fall through to the wrapped model for HuggingFace-specific attributes
            return getattr(self.model, name)

    def save_pretrained(self, *args, **kwargs):
        self.model.save_pretrained(*args, **kwargs)

class WeightedLossTrainer(Trainer):
    def __init__(self, custom_loss_weights=None, vocab_size=None, tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if custom_loss_weights is not None:
            self.criterion = create_weighted_loss(
                vocab_size=vocab_size,
                tokenizer=tokenizer,
                token_weight_dict=custom_loss_weights, 
                default_weight=1.0,
                device=self.args.device
            )
        else:
            self.criterion = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.criterion is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            logits_flat = shift_logits.view(-1, shift_logits.size(-1))
            labels_flat = shift_labels.view(-1)

            loss = self.criterion(logits_flat, labels_flat)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

def tokenize_fn(example, tokenizer, max_length=4096):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return result

def _normalize_bools(d: dict) -> dict:
    def parse(v):
        if isinstance(v, bool): return v
        if isinstance(v, str) and v.lower() == 'false': return False
        if isinstance(v, str) and v.lower() == 'true': return True
        return v
    return {k: parse(v) for k, v in d.items()}

def train(
    run_name: str, # run name that is used for saving to disk and logging to wandb
    task: str = "permutation",
    train_len: int = 30,
    task_kwargs: str = "{}", # task-level flags go here
    num_train_samples: int = 1000000,
    batch_size: int = 16, # bsize on the GPU
    target_bsize: int = 256, # target bsize. We add number of required gradient accumulation steps, if bsize < target_bsize
    lr: float = 3e-4,
    minimum_lr: float = 0.1, # fraction of the max learning rate to decay to by the end of training (cosine scheduling)
    seed: int = 42,
    warmup_ratio: float = 0.05,
    wdecay: float = 0.1,
    output_dir: str = str(MODELS_OUT_BASE_PATH),
    random_pos: bool = False, # applies a random offset in positional encodings. For APE only
    positional_encodings: str = "ape",
    resume_from_path: str = None, # pass a path to an existing model to resume training from there
    abort_step: int = None, # step at which to check eval loss and potentially abort a bad run
    abort_eval_cutoff: float = None, # max allowed eval loss at abort_step; if higher, training aborts
    best_ckpt: bool = False, # if True we save the best model based on the ood-val performance. If False, we only save the final model
):
    # Capture all arguments passed to train()
    exp_config = locals().copy()

    if not torch.distributed.is_initialized():
        if torch.cuda.is_available():
            # Extra sanity: actually try to use the GPU, not just detect it
            try:
                torch.zeros(1).cuda()
                n = torch.cuda.device_count()
                assert n > 0, "CUDA reported available but found 0 devices"
                print(f"Using nccl backend ({n} GPUs)")
                backend = "nccl"
            except Exception as e:
                raise RuntimeError(
                    f"CUDA reported as available but failed to initialize: {e}"
                ) from e
        else:
            raise RuntimeError(
                f"ERROR: CUDA not available"
            )

        torch.distributed.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=60)
        )

    task_config = TASK_REGISTRY.get(task)
    weighted_tokens = task_config.loss_weights if task_config else {}

    set_seed(seed)
    disable_caching()
    random.seed(seed)

    global_rank = int(os.environ.get("RANK", -1))
    print(f"global rank: {global_rank}")
    local_rank = int(os.environ.get("LOCAL_RANK", -1)) # local rank is specific to the node the script runs on.. maybe need this in the future
    is_main_process = global_rank <= 0

    if isinstance(task_kwargs, str):
        task_kwargs = json.loads(task_kwargs)
    elif not isinstance(task_kwargs, dict):
        raise ValueError(f"Invalid task_kwargs. Expected dict or str, got {type(task_kwargs)}")
    task_kwargs = _normalize_bools(task_kwargs)
    exp_config["task_kwargs"] = task_kwargs # Update with the parsed dict

    if resume_from_path:
        run_name = f"{run_name}_continued"
        exp_config["run_name"] = run_name
        config_path = os.path.join(resume_from_path, "exp_config.json")
        if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        old_config = json.load(f)
                    
                    # Extract the old seed and apply a large offset
                    original_seed = old_config.get("seed", seed)
                    seed = original_seed + 100
                    exp_config["seed"] = seed
                    print(f"Extracted original seed: {original_seed}")
                    print(f"Applied offset. Generating fresh data with new seed: {seed}")
                    
                except Exception as e:
                    print(f"Warning: Failed to read exp_config.json ({e}). Falling back to CLI seed: {seed}")
        else:
            print(f"Warning: No exp_config.json found in {resume_from_path}. Falling back to CLI seed: {seed}")
    full_output_dir = os.path.join(output_dir, run_name)

    if is_main_process:
        print(f"Starting Experiment: {run_name}")
        print(f"Task: {task} | Length: {train_len} | Config: {task_kwargs}")
        print("Initializing model...")

    if positional_encodings == PositionalEncodings.APE:
        model_config = AutoConfig.from_pretrained(f"{MODEL_CONFIG_BASE_PATH}/gpt2-4096-ape.json")
    elif positional_encodings == PositionalEncodings.APE_SMALL:
        model_config = AutoConfig.from_pretrained(f"{MODEL_CONFIG_BASE_PATH}/gpt2-4096-small.json")
    elif positional_encodings == PositionalEncodings.APE_XSMALL:
        model_config = AutoConfig.from_pretrained(f"{MODEL_CONFIG_BASE_PATH}/gpt2-4096-xsmall.json")
    else:
        raise ValueError(f"Unknown positional encoding {positional_encodings}. Available {[t.value for t in PositionalEncodings]}")

    if resume_from_path and os.path.exists(resume_from_path):
        if is_main_process:
            print(f"Resuming training. Loading model and tokenizer from: {resume_from_path}")
 
        # Load the exact tokenizer saved alongside the model weights
        tokenizer = AlgorithmicTaskTokenizer.from_pretrained(resume_from_path)
        model = AutoModelForCausalLM.from_pretrained(resume_from_path, torch_dtype=torch.float32)

    else:
        if is_main_process:
            print("Initializing fresh model and loading base tokenizer from config...")
            
        tokenizer_name = TASK_TO_TOKENIZER_DIR.get(task, str(task))
        tokenizer_path = os.path.join(TOKENIZER_BASE_PATH, tokenizer_name)
        
        if not os.path.exists(tokenizer_path):
            raise ValueError(f"Tokenizer not found for task {task} at {tokenizer_path}.")
            
        tokenizer = AlgorithmicTaskTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_config(model_config, torch_dtype=torch.float32)

    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.bos_token_id = tokenizer.bos_token_id
    model_config.eos_token_id = tokenizer.eos_token_id
    
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    model.config.vocab_size = model.get_input_embeddings().weight.shape[0]

    # unused. Positional dropout does not seem to add anything
    #if (positional_encodings == PositionalEncodings.APE or positional_encodings == PositionalEncodings.APE_SMALL) and pos_dropout_prob > 0.0:
    #    model = PosDropoutGPT2(model,tokenizer,pos_dropout_prob)

    print(f"Total params: {sum(p.numel() for p in model.parameters())}")
    print(f"Embedding Matrix: {model.get_input_embeddings().weight.shape}")
    data_path_train = Path(f"{DATA_BASE_PATH}/train_{run_name}.hf")
    data_path_id_val_raw = Path(f"{DATA_BASE_PATH}/id_val_raw_{run_name}.hf")
    data_path_val = Path(f"{DATA_BASE_PATH}/val_{run_name}.hf")
    data_path_val_raw = Path(f"{DATA_BASE_PATH}/val_raw_{run_name}.hf")
    if is_main_process:
        print(f"Generating {num_train_samples} samples...")
        train_dataset_raw = load_split(
            task, "train",
            num_samples=num_train_samples,
            min_length=1,
            max_length=train_len,
            seed=seed,
            save_to_path=data_path_train,
            **task_kwargs,
        )

        id_val_dataset_raw = load_split(
            task, "val",
            num_samples=VAL_SAMPLES,
            min_length=1,
            max_length=train_len,
            seed=seed + 1337,
            save_to_path=data_path_val,
            **task_kwargs,
        )

        ood_val_dataset_raw = load_split(
            task, "val",
            num_samples=VAL_SAMPLES,
            min_length=train_len + 1,
            max_length=int(1.8 * train_len),
            seed=seed + 1337,
            save_to_path=data_path_val,
            **task_kwargs,
        )

        num_proc = min(16, os.cpu_count() // 2)
        train_dataset_tokenized = train_dataset_raw.map(tokenize_fn, fn_kwargs={"tokenizer": tokenizer}, batched=True, num_proc = num_proc, remove_columns=["text"])
        ood_val_dataset_tokenized = ood_val_dataset_raw.map(tokenize_fn, fn_kwargs={"tokenizer": tokenizer}, batched=True, num_proc = num_proc, remove_columns=["text"])

        train_dataset_tokenized.save_to_disk(str(data_path_train))
        ood_val_dataset_tokenized.save_to_disk(str(data_path_val))
        ood_val_dataset_raw.save_to_disk(str(data_path_val_raw))
        id_val_dataset_raw.save_to_disk(str(data_path_id_val_raw))

    # wait for the main process to finish generating/tokenizing the dataset
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    train_dataset = load_from_disk(str(data_path_train))
    val_dataset = load_from_disk(str(data_path_val))
    ood_val_dataset_raw = load_from_disk(str(data_path_val_raw))
    id_val_dataset_raw = load_from_disk(str(data_path_id_val_raw))
    print(f"Loaded columns: {train_dataset.column_names}")
    
    if is_main_process:
        print("SANITY CHECK: EXACT TRAINING TENSOR (DECODED)")
        sample_ids = train_dataset[0]["input_ids"]
        raw_decoded = tokenizer.decode(sample_ids, skip_special_tokens=False)
        print(repr(raw_decoded)) # repr() reveals hidden newlines and spaces

    # We want to calculate loss only on the CoT, not the task description
    base_collator = DataCollatorForCompletionOnlyLM(
        response_template=TRACE_TOKEN, 
        tokenizer=tokenizer, 
        mlm=False
    )

    if random_pos:
        collator = RandomOffsetCollator(
            base_collator,
            model_config.n_positions,
            seed=seed
        )
    else:
        collator = base_collator

    accuracy_callback = BatchedAccuracyCallback(
        tokenizer=tokenizer,
        eval_dataset=ood_val_dataset_raw,
        id_eval_dataset=id_val_dataset_raw,
        task=task,
        batch_size=EVAL_BATCH_SIZE,
        max_eval_samples=MAX_EVAL_SAMPLES,
    )
    best_ckpt_callback = BestOODCheckpointCallback(metric_name="eval_ood_accuracy")

    train_instances_str = f"{num_train_samples/1000}k"

    # avoid duplicate logging from multiple processes
    if is_main_process:
        os.environ["WANDB_PROJECT"] = f"cot_{task}"
        os.environ["WANDB_TAGS"] = f"{positional_encodings},{train_instances_str}"
        os.environ["WANDB_NAME"] = run_name
        # Explicitly surface the key wandb needs
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        else:
            raise RuntimeError("WANDB_API_KEY not found in environment variables.")

    if torch.distributed.is_initialized():
        num_gpus = torch.distributed.get_world_size()
    else:
        num_gpus = 1

    target_bsize = target_bsize // batch_size
    gradient_accumulation_steps = max(1, target_bsize // num_gpus)

    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=full_output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ddp_find_unused_parameters=False,
        ddp_backend=backend,
        num_train_epochs=1,
        learning_rate=lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": minimum_lr},
        warmup_ratio=warmup_ratio,
        weight_decay=wdecay,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        per_device_eval_batch_size=batch_size,
        save_strategy="no",
        #save_steps=EVAL_STEPS,
        #save_total_limit=1,
        report_to="wandb" if is_main_process else "none",
        dataloader_num_workers=DATA_LOADER_NUM_WORKERS,
        remove_unused_columns=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    callback_list = [accuracy_callback, WandbConfigCallback(exp_config)]
    if best_ckpt:
        callback_list.append(best_ckpt_callback)
    if abort_step and abort_eval_cutoff:
        callback_list.append(EarlyAbortCallback(abort_step=abort_step, max_eval_loss=abort_eval_cutoff))

    trainer = WeightedLossTrainer(
        custom_loss_weights=weighted_tokens,
        vocab_size=model.config.vocab_size,
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callback_list
    )

    print(f"Starting training on process {global_rank}...")
    trainer.train()
    if is_main_process:
        if not best_ckpt:
            model.save_pretrained(full_output_dir)
        tokenizer.save_pretrained(full_output_dir)
        
        with open(os.path.join(full_output_dir, "exp_config.json"), "w") as f:
            json.dump(exp_config, f, indent=2)

        print(f"Done! Model saved to {full_output_dir}")

if __name__ == "__main__":
    fire.Fire(train)