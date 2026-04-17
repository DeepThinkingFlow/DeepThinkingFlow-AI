# Skill Compliance

Use this reference when the user asks whether DeepThinkingFlow actually learned a behavior, whether `model.safetensors` contains a skill, or whether runtime prompting is enough to call something production.

## What Runtime Can Do

- Inject a system prompt, profile, and chat template at inference time.
- Shape visible answer style, structure, and verbosity.
- Keep `analysis` optional and sanitized while defaulting end-user output to `final`.
- Make the current runtime look more compliant without changing the underlying weights.

## What Runtime Cannot Do

- It cannot rewrite the knowledge already stored in `model.safetensors`.
- It cannot honestly be described as weight-level learning.
- It cannot prove the model would behave the same way once the prompt, profile, or wrapper script is removed.
- It cannot justify claims like "the raw checkpoint now contains this skill."

## What SFT Does

- SFT examples define the target behavior clearly.
- SFT datasets make behavior measurable, repeatable, and trainable.
- SFT artifacts are the bridge between prompt ideas and learned behavior.

Important:

- A dataset alone is still not learned behavior.
- Before training runs, SFT is `training-ready`, not `weight-level`.

## What LoRA Or QLoRA Changes

- LoRA or QLoRA is the first stage where you can start talking about learned behavior.
- The learned behavior lives in the adapter artifact, not magically inside the untouched base checkpoint.
- To claim this honestly, you need the adapter weights, config, training logs, and eval results.

## What Counts As Weight-Level Adherence

Only these can support a weight-level claim:

- a merged checkpoint created after training
- a newly trained checkpoint
- reproducible train artifacts plus eval evidence that map to that resulting weight artifact

Anything else is weaker and should be labeled accordingly.

## Red-Flag Claims To Avoid

- "I edited `SKILL.md`, so `model.safetensors` learned the skill."
- "I changed the runtime prompt, so the weights are now more obedient."
- "I manually touched `model.safetensors`, therefore the model understands the new policy."
- "The model now has Opus internals."
- "Showing full chain-of-thought is proof of stronger reasoning."

## Claim Labels

| Label | Meaning |
| --- | --- |
| `runtime-only` | Behavior comes from prompt, wrapper, profile, or output filtering at inference time. |
| `training-ready` | Datasets, configs, and eval scaffolding exist, but no learned artifact is proven yet. |
| `learned-only-after-training` | The claim becomes honest only after LoRA/QLoRA or other training finishes and passes eval. |
| `verified-on-current-file` | The claim is directly verified against the current local file or artifact. |

## Canonical Dataset Buckets

The skill-compliance SFT dataset in this repo should stay inside exactly these buckets:

- `reject-false-weight-claim`
- `runtime-vs-learned`
- `short-analysis-no-cot`
- `deep-style-without-fake-internals`

If a new example does not fit one of these buckets, treat that as a design review moment rather than silently widening the taxonomy.

## Short Decision Rule

- If the behavior disappears when you remove the wrapper or prompt, it was `runtime-only`.
- If you only created examples or configs, it is `training-ready`.
- If you trained an adapter and can show the artifact plus eval, it is `learned-only-after-training`.
- If you are describing what is literally inside `model.safetensors`, only say what was `verified-on-current-file`.
