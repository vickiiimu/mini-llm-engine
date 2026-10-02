# Mini LLM Engine

An educational LLM inference engine built to understand and measure
the systems behind modern LLM serving.

## Status

Repository setup. Inference implementation has not started.

## Planned milestones

- [ ] Manual autoregressive generation baseline
- [ ] KV caching and prefill/decode measurement
- [ ] Continuous batching
- [ ] Block-based KV-cache allocation
- [ ] Benchmarks against Hugging Face and vLLM

## First milestone

Generate text for one request without using model.generate().
Record prompt length, generated tokens, latency, throughput,
and peak GPU memory.