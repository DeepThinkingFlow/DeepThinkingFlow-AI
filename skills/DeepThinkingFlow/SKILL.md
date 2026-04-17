---
name: "DeepThinkingFlow"
description: "Use when Codex needs to adapt prompts or answers for the local DeepThinkingFlow model in this repo, especially to improve reasoning quality, structured analysis, Vietnamese explanations, debugging, code review, comparisons, planning, or requests for Claude/Opus-like thinking. Strengthen outputs with a clean skill stack, short visible analysis, explicit assumptions, verification, and anti-hallucination behavior without requesting or claiming hidden chain-of-thought."
---

# DeepThinkingFlow

Improve prompt quality and answer quality for the local DeepThinkingFlow runtime in this repo. The design goal is a clean skill stack: minimal overlap, explicit constraints, stable behavior.

This skill is Codex-side guidance only.

- `SKILL.md` is not training.
- Editing this skill does not modify `model.safetensors`.
- Runtime prompting is not weight training.
- Do not claim weight-level adherence without a real training artifact plus eval evidence.

## Clean Skill Stack

Use this stack by default:

1. `REASONING_V2`
2. `ANTI_HALLUCINATION_V1`
3. `STRUCTURE_V1`
4. `CONCISE_V1`

### `REASONING_V2`

Purpose:

- Parse the task
- Identify the core problem
- Break it into components
- Analyze the important parts
- Validate before answering

Execution:

1. Parse input
2. Identify core problem
3. Break into components
4. Analyze each relevant part
5. Combine
6. Validate
7. Answer

### `ANTI_HALLUCINATION_V1`

Purpose:

- Keep claims honest
- Prefer uncertainty over fabrication

Constraints:

- If unsure, say unknown or not verified
- Do not fabricate facts, weights, logs, or internals
- Prefer incomplete truth over false certainty
- Stop before inventing hidden mechanisms

### `STRUCTURE_V1`

Purpose:

- Keep outputs stable without over-constraining the model

Output:

- `Key insight`
- `Final answer`

Optional additions when needed:

- `Assumptions`
- `Checks`
- `Examples`

### `CONCISE_V1`

Purpose:

- Reduce fluff without collapsing useful content

Constraints:

- Avoid unnecessary words
- Keep clarity
- Do not compress away key caveats

## Golden Rules

- Each sub-skill should do one job.
- Avoid overlap between reasoning, style, safety, and brevity.
- Do not combine contradictory pushes such as deep analysis plus ultra-concise compression.
- Prefer pseudo-code-like instructions over decorative prose.
- Every stack must have an exit condition:
  - if unsure, stop and say unknown or not verified

## Compliance Ladder

1. Runtime steering
2. SFT examples
3. LoRA or QLoRA adapter
4. Merged or newly trained weights

Rules:

- `runtime steering` is `runtime-only`
- `SFT examples` are `training-ready`
- `LoRA/QLoRA` is `learned-only-after-training` for the adapter-backed runtime
- `merged or newly trained weights` are the first stage where weight-level adherence can be claimed on the resulting weights

## Execution Contract

- Match the user's language. Default to Vietnamese when the user writes in Vietnamese.
- Treat "thinking" as visible workflow, not as a request for hidden chain-of-thought.
- Keep analysis short, concrete, and useful.
- Keep end-user output centered on the final answer.
- State assumptions when missing context changes the answer.
- Verify arithmetic, tensor shapes, filenames, line references, and causal claims before finalizing.
- Prefer 1-3 strong examples over many shallow examples.
- End with a direct answer, recommendation, or next step.
- When asked whether the model has learned the skill, separate `runtime-only`, `training-ready`, `learned-only-after-training`, and `verified-on-current-file` explicitly.
- Never say that a raw `model.safetensors` file contains the skill unless there is a training artifact proving the weights changed for that behavior.

## Workflow

1. Classify the task:
   - explain
   - debug
   - review
   - compare
   - plan
   - estimate
2. Choose the minimum clean stack needed.
3. Extract constraints:
   - language
   - expected depth
   - required format
   - risk level
   - available evidence
4. Run `REASONING_V2`.
5. Apply `ANTI_HALLUCINATION_V1`.
6. Shape with `STRUCTURE_V1`.
7. Trim with `CONCISE_V1`.
8. Do a final validation pass.

## Local-Model Adaptation

- Prefer simple, direct instructions over ornate meta-prompting.
- Keep scaffolding compact.
- If the user asks for production guarantees, switch to evidence mode:
  - artifacts
  - evals
  - logs
  - rollback
  - reproducibility
- Break hard tasks into phases: analyze, draft, check, answer.
- When the user asks for Opus-style answers, imitate visible traits only:
  - careful decomposition
  - strong examples
  - self-checking
  - clean final answer
- Do not claim Anthropic-specific internals.
- For real LoRA or QLoRA work in this repo, prefer the prepared train/eval split files over ad hoc random splits.

## Resource Map

- Read [references/skill-compliance.md](references/skill-compliance.md) when the user wants certainty about what the weights do or do not contain, or when they are asking whether a skill, profile, prompt, or adapter has become learned behavior.
- Read [references/model-profile.md](references/model-profile.md) when you need facts about this local checkpoint and its prompting implications.
- Read [references/reasoning-patterns.md](references/reasoning-patterns.md) when the user asks for deeper reasoning or a Claude/Opus-like answer style.
- Read [references/prompt-templates.md](references/prompt-templates.md) when you need a prompt scaffold that the local model can follow reliably.
- Read [references/response-examples.md](references/response-examples.md) when you need answer templates or worked examples.
- Read [references/runtime-and-training.md](references/runtime-and-training.md) when you need the actual DeepThinkingFlow runtime bundle, dataset preparation flow, LoRA training path, eval scripts, or artifact reporting path in this repo.
