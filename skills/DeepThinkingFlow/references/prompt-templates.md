# Prompt Templates

Use these templates when the local DeepThinkingFlow model needs tighter structure. Keep only the tags that help. Replace placeholders directly.

## 1. General reasoning

```text
<task>
<describe the user's goal in one sentence>
</task>

<language>
Vietnamese
</language>

<reasoning_policy>
Think carefully, but keep the visible analysis short and useful.
State assumptions explicitly when information is missing.
Do not expose hidden chain-of-thought.
</reasoning_policy>

<output_format>
Goal
Assumptions
Analysis
Answer
Examples
Checks
</output_format>
```

## 2. Debugging

```text
<task>
Debug this issue.
</task>

<inputs>
<logs, code, symptoms, environment>
</inputs>

<instructions>
List the most likely causes first.
Then list the checks to run in order.
Then give the most probable fix.
Keep reasoning concise and practical.
</instructions>

<output_format>
Symptoms
Likely causes
Checks
Most probable fix
Fallback fix
</output_format>
```

## 3. Code review

```text
<task>
Review this change like a careful senior engineer.
</task>

<instructions>
Find bugs, regressions, missing tests, and unclear assumptions.
Put findings first.
Keep summaries brief.
Use file and line references when available.
</instructions>

<output_format>
Findings
Open questions
Short summary
</output_format>
```

## 4. Comparison and recommendation

```text
<task>
Compare these options and recommend one.
</task>

<instructions>
Compare on 3-5 concrete criteria.
Recommend one option first.
Explain the main tradeoff and one failure mode.
</instructions>

<output_format>
Recommendation
Comparison
Tradeoff
Failure mode
Example scenario
</output_format>
```

## 5. Teaching / explanation

```text
<task>
Explain this topic to a learner.
</task>

<instructions>
Start simple.
Then give one worked example.
Then list common mistakes.
Avoid jargon unless you define it.
</instructions>

<output_format>
Short explanation
Worked example
Common mistakes
Practical takeaway
</output_format>
```

## 6. Plan / migration / rollout

```text
<task>
Create a practical implementation plan.
</task>

<instructions>
Split the work into phases.
Include validation and rollback.
Call out the main risk.
Keep the plan concrete.
</instructions>

<output_format>
Goal
Phases
Validation
Rollback
Main risk
</output_format>
```
