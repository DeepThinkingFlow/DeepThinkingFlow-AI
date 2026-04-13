# DeepThinkingFlow Model Profile

This profile is derived from `original/model.safetensors`, `original/config.json`, and `original/dtypes.json` in this repo.
DeepThinkingFlow currently runs on top of an upstream GPT-OSS-compatible checkpoint, so the raw tensor and architecture facts below still describe that compatibility layer.

## Snapshot

- File size: about 12.82 GiB
- Architecture: transformer with MoE MLP blocks
- Layers: 24
- Hidden size: 2880
- Vocab size: 201088
- Attention heads: 64 query heads
- KV heads: 8
- Head dim: 64
- Experts per layer: 32
- Experts active per token: 4
- Sliding window: 128
- Initial context length: 4096

## Weight Layout

- `embedding.weight` and `unembedding.weight` are BF16
- Attention weights are BF16
- MoE MLP expert weights are stored as packed FP4 blocks with UE8 scales
- The safetensors header itself only exposes `BF16` and `U8`; `dtypes.json` clarifies that the `U8` tensors represent packed `FP4` blocks and `UE8` scales

## Practical Prompting Implications

- Expect better results from explicit structure than from vague "think harder" instructions
- Prefer compact multi-step prompting over long, theatrical scratchpads
- Provide the desired output shape when quality matters
- Ask for assumptions and checks explicitly on technical tasks
- Use a few well-chosen examples when the format is important
- Avoid stacking too many competing constraints in one prompt

## Reasoning Implications

- Treat the model as capable but smaller than frontier reasoning models in effective per-token capacity
- Favor decomposition, verification, and intermediate summaries
- For complex tasks, ask for:
  - the goal
  - the assumptions
  - the key analysis
  - the final answer
  - one or more examples
- For coding or debugging, ask for likely causes first, then concrete checks, then the fix

## Notes

- A rough inference from tensor shapes suggests about 21.5B total logical parameters if packed FP4 weights are expanded conceptually
- A rough active-per-token estimate is about 4.19B parameters because only 4 of 32 experts are selected per layer
- Treat those parameter counts as engineering estimates, not official vendor metadata
- This skill file does not enforce model behavior by itself. For real runtime behavior, use the Transformers integration and DeepThinkingFlow behavior bundle in this repo.
