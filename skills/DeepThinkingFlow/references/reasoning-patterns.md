# Reasoning Patterns

Use these patterns to emulate the observable strengths of strong "thinking" assistants while staying compatible with DeepThinkingFlow.

## What To Imitate

- Give a clean restatement of the task
- Separate assumptions from confirmed facts
- Decompose hard tasks into small steps
- Check the result before finalizing
- Use examples to lock tone, format, or depth
- Keep the final answer readable and action-oriented

## What To Avoid

- Do not claim access to hidden chain-of-thought
- Do not dump a long scratchpad when a short analysis would do
- Do not overfit to one example if the task is broad
- Do not confuse certainty with style; cautious language is better than fake confidence

## Prompt Moves That Travel Well

- Ask for a short analysis before the final answer
- Ask for assumptions explicitly when the prompt is underspecified
- Ask for a self-check or verification pass
- Ask for one worked example when teaching
- Ask for comparison criteria before asking for a recommendation
- Ask for failure modes when the user is making a decision

## Suggested Prompt Templates

Use prompts like these when adapting for the local model:

```text
Answer in Vietnamese. Restate the goal, list the key assumptions, give a brief analysis, then give the final answer and one concrete example.
```

```text
Compare the options using 3-5 criteria, recommend one, explain why, and include one failure mode or tradeoff I should watch for.
```

```text
Debug this step by step: likely causes first, then the checks I should run, then the most probable fix. Keep the reasoning concise.
```

```text
Teach this like a strong reasoning assistant: simple explanation first, then a worked example, then common mistakes.
```

## Source-Inspired Guidance

These patterns are adapted from official Anthropic documentation on Claude model prompting and extended thinking. The relevant ideas are:

- Use clear, direct instructions
- Give examples when you need consistent format or style
- Separate source material from instructions cleanly
- Ask for verification on complex tasks
- Use "thinking" for hard problems, but keep the user-facing answer focused

Use those ideas as style guidance, not as a request to mimic private internals or vendor-specific APIs.
