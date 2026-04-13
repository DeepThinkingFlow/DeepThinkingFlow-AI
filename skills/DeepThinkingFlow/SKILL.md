---
name: "DeepThinkingFlow"
description: "Use when Codex needs to adapt prompts or answers for the local DeepThinkingFlow model in this repo, especially to improve reasoning quality, structured analysis, Vietnamese explanations, debugging, code review, comparisons, planning, or requests for Claude/Opus-like thinking. Strengthen outputs with short visible analysis, explicit assumptions, verification, and example-led formatting without requesting or claiming hidden chain-of-thought."
---

# DeepThinkingFlow

Improve prompt quality and answer quality for the local DeepThinkingFlow runtime in this repo. Optimize for observable reasoning quality: clear assumptions, compact analysis, verification, and strong final answers.

This skill is Codex-side guidance only. It does not modify model weights, does not guarantee runtime enforcement, and does not replace a real Transformers/vLLM/Ollama integration.

## Quick start

- Load [references/model-profile.md](references/model-profile.md) for checkpoint facts and prompting implications.
- Load [references/reasoning-patterns.md](references/reasoning-patterns.md) for reasoning behavior.
- Load [references/prompt-templates.md](references/prompt-templates.md) when rewriting prompts or building a reusable system prompt.
- Load [references/response-examples.md](references/response-examples.md) when shaping the final answer.
- Load [references/runtime-and-training.md](references/runtime-and-training.md) when the user wants a real runtime integration, LoRA/SFT workflow, or production hardening path with fixed train/eval splits.

## Execution contract

- Match the user's language. Default to Vietnamese when the user writes in Vietnamese.
- Treat "thinking" as visible workflow, not as a request for hidden chain-of-thought.
- Keep analysis short, concrete, and useful.
- State assumptions when missing context changes the answer.
- Verify arithmetic, tensor shapes, filenames, line references, and causal claims before finalizing.
- Prefer 1-3 strong examples over many shallow examples.
- End with a direct answer, recommendation, or next step.

## Workflow

1. Classify the task:
   - explain
   - debug
   - review
   - compare
   - plan
   - estimate
2. Extract constraints:
   - language
   - expected depth
   - required format
   - risk level
   - available evidence
3. Choose response depth:
   - Quick: answer first, add one small example only if it helps
   - Standard: assumptions, short analysis, final answer, one or two examples
   - Deep: plan, tradeoffs, verification, final answer, two to four examples
4. Select one prompt scaffold from [references/prompt-templates.md](references/prompt-templates.md) if prompt rewriting is needed.
5. Select one answer pattern from [references/response-examples.md](references/response-examples.md).
6. Run a final check for missing caveats, weak claims, and unsupported certainty.

## Local-Model Adaptation

- Prefer simple, direct instructions over ornate meta-prompting.
- Keep scaffolding compact. Local models benefit from structure, but too many rules can reduce answer quality.
- Ask for comparison, critique, or verification explicitly when needed.
- Break hard tasks into phases: analyze, draft, check, answer.
- When the task is long or technical, provide a target format to imitate.
- When the user asks for "Opus-style" answers, imitate visible traits only: careful decomposition, strong examples, self-checking, and a clean final answer. Do not claim Anthropic-specific internals.
- For real LoRA or QLoRA work in this repo, prefer the prepared train/eval split files over ad hoc random splits so before/after comparisons stay stable.

## Output contract

- Use `Goal / Assumptions / Analysis / Answer / Examples / Checks` for complex tasks.
- Skip `Analysis` entirely for trivial requests.
- Put findings first for reviews and bug triage.
- Put the recommendation first for comparisons and decisions.
- Put the worked example close to the explanation for teaching tasks.

## Preferred answer skeleton

Use this only when it helps. Do not force it on trivial tasks.

```text
Goal: <one-sentence restatement>
Assumptions: <only if needed>
Analysis: <short visible reasoning>
Answer: <direct answer or recommendation>
Examples: <1-3 concrete examples>
Checks: <verification, caveat, or next step>
```

## Resource map

- Read [references/model-profile.md](references/model-profile.md) when you need facts about this local checkpoint and its prompting implications.
- Read [references/reasoning-patterns.md](references/reasoning-patterns.md) when the user asks for deeper reasoning, "thinking mode", or a Claude/Opus-like answer style.
- Read [references/prompt-templates.md](references/prompt-templates.md) when you need a prompt scaffold that the local model can follow reliably.
- Read [references/response-examples.md](references/response-examples.md) when you need answer templates or worked examples for debugging, review, explanation, planning, or comparison tasks.
- Read [references/runtime-and-training.md](references/runtime-and-training.md) when you need to connect the Codex-side guidance in this skill to the actual DeepThinkingFlow runtime bundle, dataset preparation, LoRA training, or eval scripts in this repo.
