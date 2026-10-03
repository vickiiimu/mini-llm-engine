import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# for CPU

model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cpu")
model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"Input IDS: {inputs['input_ids']}")
print(f"Input shape: {inputs['input_ids'].shape}")

with torch.inference_mode():
    outputs = model(**inputs, use_cache=False)

print(f"Logits shape: {outputs.logits.shape}")

next_token_ids = outputs.logits[:, -1:].argmax(-1)

print(f"Next token ID: {next_token_ids.item()}")
print(f"Next token: {tokenizer.decode(next_token_ids.item())}")