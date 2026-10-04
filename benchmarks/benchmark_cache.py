import time
import torch
from statistics import median
from transformers import AutoTokenizer, AutoModelForCausalLM
from engine.model_runner import greedy_decode

model_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cpu")
model.eval()

prompt = "The capital city of France is" * 10
input = tokenizer(prompt, return_tensors="pt")
input_ids = input.input_ids

max_new_tokens = 10
num_trials = 5

# warm-up cache
for use_cache in [True, False]:
    greedy_decode(model, input_ids, max_new_tokens, use_cache)


times = {
    "cached": [],
    "uncached": [],
}

for trial in range(1, num_trials + 1):
    # switch order of testing to disentangle computer conditions
    modes = [True, False] if trial % 2 else [False, True]
    results = {}

    for use_cache in modes:
        mode = "cached" if use_cache else "uncached"

        start = time.perf_counter()
        generated_ids = greedy_decode(model, input_ids, max_new_tokens, use_cache)

        t_elapsed = time.perf_counter() - start

        times[mode].append(t_elapsed)
        results[mode] = generated_ids

    assert torch.equal(results["cached"], results["uncached"])

cached_median = median(times["cached"])
uncached_median = median(times["uncached"])
speedup = uncached_median/cached_median

print(f"Prompt tokens: {input_ids.shape[1]}")
print(f"Generated tokens: {max_new_tokens}")
print(f"Uncached median: {uncached_median * 1000:.1f} ms")
print(f"Cached median: {cached_median * 1000:.1f} ms")
print(f"Speedup: {speedup:.2f}x")