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

# calculate actual KV cache size
cache_bytes = 0

for layer_cache in past_key_values.layers:
    k_bytes = layer_cache.keys.numel() * layer_cache.keys.element_size()
    v_bytes = layer_cache.values.numel() * layer_cache.values.element_size()

    cache_bytes += k_bytes + v_bytes

print(
    f"Cached tokens: {past_key_values.get_seq_length()} | "
    f"KV cache size: {cache_bytes} B | "
    f"KV cache size: {cache_bytes / 1024:.2f} KiB"
)



# calculate expected KV cache size
num_layers = model.config.num_hidden_layers
num_kv_heads = model.config.num_key_value_heads

head_dim = model.config.hidden_size // model.config.num_attention_heads

batch_size = generated_ids.shape[0]
cached_tokens = past_key_values.get_seq_length()
bytes_per_element = past_key_values.layers[0].keys.element_size()

bytes_per_token = (
    2 # K and V vector
    * num_layers
    * num_kv_heads
    * head_dim
    * bytes_per_element
)

expected_cache_bytes = batch_size * cached_tokens * bytes_per_token
print(f"Expected cache size: {expected_cache_bytes} B | {expected_cache_bytes / 1024:.2f} KiB")