import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cpu")
model.eval()

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


prompts = [
    "The capital of France is",
    "Hi! My name is",
    "In my free time, I like to",
    "Once upon a time"
    ]

for prompt in prompts:
    input = tokenizer(prompt, return_tensors="pt")
    input_ids = input.input_ids

    max_new_tokens = 10

    uncached_ids = greedy_decode(model, input_ids, max_new_tokens, False)

    cached_ids = greedy_decode(model, input_ids, max_new_tokens, True)

    print(f"Uncached IDS: {uncached_ids}")
    print(f"Cached IDS: {cached_ids}")

    match = torch.equal(uncached_ids, cached_ids)
    print(f"Token IDS match? {match}")

    assert match, f"Uncached and cached generation produce different token IDS for prompt: {prompt}"