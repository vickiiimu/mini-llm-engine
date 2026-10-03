import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cpu")
past_key_values = DynamicCache(config=model.config)
model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")
generated_ids = inputs['input_ids']

print(f"Input IDS: {generated_ids}")
print(f"Input shape: {generated_ids.shape}")

current_ids = generated_ids
max_new_tokens = 10


with torch.inference_mode():
    for step in range(1, max_new_tokens + 1):
        seq_len = current_ids.shape[1]
        start = time.perf_counter()

        outputs = model(current_ids, past_key_values=past_key_values, use_cache=True)

        next_token_ids = outputs.logits[:, -1:].argmax(-1)

        t_elapsed = time.perf_counter() - start

        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
        current_ids = next_token_ids

        output_text = tokenizer.decode(next_token_ids.item())

        print(
            f"Step: {step} | "
            f"Input length: {seq_len} | "
            f"Next token: {output_text} | "
            f"Cached tokens: {past_key_values.get_seq_length()} | "
            f"Elapsed time (ms): {t_elapsed * 1000}"    
        )

        if step <= 3:
            for layer_idx in range(3): #first 3 attention layers
                layer_cache = past_key_values.layers[layer_idx]

                # (batch_size, number_of_kv_heads, cached_tokens, head_dimension)
                print(
                    f"Layer {layer_idx} | "
                    f"K shape: {layer_cache.keys.shape} | "
                    f"V shape: {layer_cache.values.shape}"
                )

print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))