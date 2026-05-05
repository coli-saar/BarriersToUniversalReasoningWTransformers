import torch
import random

class RandomOffsetCollator:
    def __init__(self, base_collator, max_context_len=4096, seed=None):
        self.base_collator = base_collator
        self.max_context_len = max_context_len
        
        # Set seed per process
        if seed is not None:
            local_rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
            process_seed = seed + local_rank
            self.rng = random.Random(process_seed)
        else:
            self.rng = random.Random()

    def __call__(self, examples):
        batch = self.base_collator(examples)
        batch_size, seq_len = batch["input_ids"].shape
        
        seq_lens = torch.sum(batch["attention_mask"], dim=1)
        
        max_offsets = torch.clamp(self.max_context_len - seq_lens, min=0)
        
        start_positions = torch.tensor(
            [self.rng.randint(0, max_offsets[i].item()) for i in range(batch_size)],
            device=batch["input_ids"].device
        )
        
        # Create position_ids row by row
        position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=batch["input_ids"].device)
        
        for i in range(batch_size):
            actual_len = seq_lens[i].item()
            offset = start_positions[i].item()
            
            # Only assign positions for actual tokens
            position_ids[i, :actual_len] = torch.arange(offset, offset + actual_len, device=batch["input_ids"].device)
        
        batch["position_ids"] = position_ids

        return batch
