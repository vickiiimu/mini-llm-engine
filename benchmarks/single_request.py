import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cpu")
model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")
generated_ids = inputs['input_ids']

print(f"Input IDS: {generated_ids}")
print(f"Input shape: {generated_ids.shape}")

max_new_tokens = 10

with torch.inference_mode():
    for step in range(1, max_new_tokens + 1):
        seq_len = generated_ids.shape[1]
        start = time.perf_counter()

        outputs = model(generated_ids, use_cache=False)

        next_token_ids = outputs.logits[:, -1:].argmax(-1)

        t_elapsed = time.perf_counter() - start

        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

        output_text = tokenizer.decode(next_token_ids.item())

        print(
            f"Step: {step} | "
            f"Input length: {seq_len} | "
            f"Next token: {output_text} | "
            f"Elpased time (ms): {t_elapsed * 1000}"    
        )