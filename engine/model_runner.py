import torch
from transformers import DynamicCache

def greedy_decode(model, input_ids, max_new_tokens, use_cache):
    if use_cache:
        past_key_values = DynamicCache(config=model.config)

    generated_ids = input_ids
    current_ids = input_ids

    with torch.inference_mode():
        for step in range(1, max_new_tokens + 1):
            if use_cache:
                outputs = model(current_ids, past_key_values=past_key_values, use_cache=True)
            else:
                outputs = model(generated_ids, use_cache=False)

            next_token_ids = outputs.logits[:, -1:].argmax(-1)

            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
            current_ids = next_token_ids

    return generated_ids