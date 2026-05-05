import torch
import torch.nn as nn

def num_to_base_26(n):
    if n < 0: 
        return ""
    if n == 0:
        return 'a'
    result = []
    while n >= 0:
        result.append(chr(ord('a') + (n % 26)))
        n //= 26
        n -= 1
    return ''.join(reversed(result))

def num_to_bracket_hint(n):
    if n < 0:
        return ""
    return f"<{n}>"

def manual_generate(model, tokenizer, inputs, max_new_tokens=2048):
    curr_input_ids = inputs['input_ids']
    curr_position_ids = inputs['position_ids']
    attention_mask = inputs['attention_mask']
    
    batch_size = curr_input_ids.shape[0]
    device = curr_input_ids.device
    
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
    generated_tokens = []
    
    past_key_values = None
    
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    
    for _ in range(max_new_tokens):
        if past_key_values is None:
            model_inputs = {
                "input_ids": curr_input_ids,
                "attention_mask": attention_mask,
                "position_ids": curr_position_ids,
                "use_cache": True
            }
        else:
            model_inputs = {
                "input_ids": curr_input_ids,
                "attention_mask": attention_mask,
                "position_ids": curr_position_ids,
                "past_key_values": past_key_values,
                "use_cache": True
            }

        outputs = model(**model_inputs)
        
        logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=-1)
        
        if eos_token_id is not None:
            if pad_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            is_eos = next_tokens == eos_token_id
            unfinished_sequences = unfinished_sequences.mul((~is_eos).long())
            
        if unfinished_sequences.max() == 0:
            break
            
        next_tokens_tensor = next_tokens.unsqueeze(-1) # [B, 1]
        generated_tokens.append(next_tokens_tensor)
        
        curr_input_ids = next_tokens_tensor
        curr_position_ids = curr_position_ids[:, -1:] + 1
        
        new_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)
        
        past_key_values = outputs.past_key_values

    if generated_tokens:
        full_gen_ids = torch.cat(generated_tokens, dim=1)
    else:
        full_gen_ids = torch.tensor([], dtype=torch.long, device=device)
        
    return full_gen_ids


def create_weighted_loss(vocab_size, tokenizer, token_weight_dict, default_weight=1.0, device="cpu"):
    weights = torch.full((vocab_size,), default_weight, dtype=torch.float, device=device)
    for token_str, weight in token_weight_dict.items():
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        
        if token_id == tokenizer.unk_token_id and token_str != tokenizer.unk_token:
            print(f"WARNING: Token '{token_str}' not found in tokenizer vocab. Skipping.")
            continue
            
        weights[token_id] = weight
        print(f"Set weight for token '{token_str}' (ID {token_id}) to {weight}")

    ignore_idx = -100
    
    return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_idx)