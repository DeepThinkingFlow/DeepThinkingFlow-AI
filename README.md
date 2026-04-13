<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-GPLv3-green?style=for-the-badge&logo=gnu&logoColor=white" />
  <img src="https://img.shields.io/badge/Transformers-4.57%2B-orange?style=for-the-badge&logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.7%2B-red?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/PEFT-0.17%2B-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Tests-7%2F7%20Passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white" />
</p>

<h1 align="center">DeepThinkingFlow-AI</h1>

<p align="center">
  <strong>Runtime-Steering & SFT-Seed Stack for Structured Reasoning</strong><br/>
  <em>Bilingual (Vietnamese/English) | LoRA/QLoRA Fine-Tuning | Behavior Bundles | Heuristic Eval</em>
</p>

<p align="center">
  A self-built, end-to-end AI reasoning pipeline on top of a GPT-OSS-20B compatibility layer.<br/>
  Focused on <strong>structured reasoning</strong>, <strong>bilingual behavior steering</strong>, and <strong>adapter-based fine-tuning</strong>.
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Workflows](#workflows)
  - [Inference Workflow](#1-inference-workflow)
  - [Training Workflow](#2-training-workflow)
  - [Evaluation Workflow](#3-evaluation-workflow)
  - [Full Pipeline Workflow](#4-full-pipeline-end-to-end)
- [Behavior Bundle System](#behavior-bundle-system)
- [Model Profile](#model-profile)
- [Training Configuration](#training-configuration)
- [Testing](#testing)
- [Codex Skill Integration](#codex-skill-integration)
- [Dataset Statistics](#dataset-statistics)
- [Design Principles](#design-principles)
- [License](#license)

---

## Overview

DeepThinkingFlow is a comprehensive AI system consisting of:

| Component | Description |
|---|---|
| **Runtime Steering** | Controls model behavior through behavior bundles (system prompt + profile) **without modifying weights** |
| **SFT Seed Data** | Bilingual Vietnamese/English training dataset in "harmony" format for supervised fine-tuning |
| **LoRA/QLoRA Training** | Complete adapter training pipeline with fixed train/eval splits, early stopping, gradient checkpointing |
| **Multi-turn Chat** | Interactive terminal chat with conversation history and dynamic reasoning effort switching |
| **Heuristic Evaluation** | Scores outputs against a trait checklist and rubric rules |
| **Unified CLI** | Single entry point for all scripts via `deepthinkingflow_cli.py` |

### Key Features

- **Bilingual (Vietnamese/English)** -- defaults to Vietnamese when the user writes in Vietnamese
- **Behavior Bundles** -- cleanly separates system prompt, profile, SFT data, and eval cases
- **3 Reasoning Levels** -- `low`, `medium`, `high` -- switchable mid-session
- **Structured Output** -- Goal, Assumptions, Analysis, Answer, Examples, Checks
- **No hidden chain-of-thought claims** -- only visible analysis when opted in
- **7/7 smoke tests passing** -- covers CLI, runtime helpers, chat flow, prompt rendering, one-shot generation

---

## Architecture

```
+---------------------------------------------------------------------+
|                        DeepThinkingFlow Stack                        |
+---------------------------------------------------------------------+
|                                                                      |
|  +---------------+     +--------------------+     +-----------------+|
|  |   User CLI    |---->|  Unified CLI       |---->|   Sub-scripts   ||
|  |  (terminal)   |     |  deepthinkingflow  |     |   (12 scripts)  ||
|  +---------------+     |  _cli.py           |     +--------+--------+|
|                         +--------------------+              |         |
|                                                             v         |
|  +----------------------------------------------------------------+  |
|  |                    deepthinkingflow_runtime.py                  |  |
|  |  +--------------+ +----------------+ +------------------------+|  |
|  |  | Model Loader | | Prompt Render  | | Response Extractor     ||  |
|  |  | + Memory     | | + Chat Templ   | | (analysis / final)     ||  |
|  |  |   Check      | |                | |                        ||  |
|  |  +-------+------+ +-------+--------+ +-----------+------------+|  |
|  +----------+----------------+---------------------------+---------+  |
|             |                |                           |            |
|             v                v                           v            |
|  +----------------------------------------------------------------+  |
|  |              HuggingFace Transformers Runtime                   |  |
|  |  AutoModelForCausalLM + AutoTokenizer + chat_template           |  |
|  +---------------------------------+------------------------------+  |
|                                    |                                  |
|            +-----------------------+-----------------------+          |
|            v                       v                       v          |
|  +--------------+     +-------------------+     +------------------+ |
|  |  Behavior    |     |  Model Weights    |     |  LoRA Adapters   | |
|  |  Bundle      |     |  (safetensors)    |     |  (PEFT)          | |
|  +--------------+     +-------------------+     +------------------+ |
|                                                                      |
+----------------------------------------------------------------------+
```

### Inference Flow

```
User Input --> Behavior Bundle (system_prompt.txt) --> Chat Template Rendering
                                                             |
                                                             v
                                                    Tokenizer.apply_chat_template()
                                                             |
                                                             v
                                                    Model.generate()
                                                             |
                                                             v
                                                    Decode Completion
                                                             |
                                              +--------------+--------------+
                                              v                             v
                                    extract_analysis_text()       extract_final_text()
                                     (hidden by default)          (shown to user)
```

### Training Flow

```
harmony_sft_vi.jsonl --> prepare_harmony_sft_dataset.py --> train.jsonl + eval.jsonl
                                                                   |
                                                                   v
config.example.json --> train_transformers_deepthinkingflow_lora.py
                              |
                              +-- Validate config + dataset
                              +-- Load tokenizer + encode examples
                              +-- Load base model (bf16 / 4-bit)
                              +-- Apply LoraConfig (PEFT)
                              +-- Train with HF Trainer
                              +-- EarlyStoppingCallback
                              +-- Save adapter to out/
                              +-- (Optional) Merge adapter back
```

---

## Project Structure

```
deepthinkingflow/
|
|-- README.md                             <- You are reading this file
|-- LICENSE                               <- GNU General Public License v3
|-- .gitignore                            <- Ignores model weights & training outputs
|-- requirements-transformers.txt         <- Dependencies for inference
|-- requirements-train-gpt-oss.txt        <- Dependencies for training (torch, peft, etc.)
|
|-- behavior/                             <- Behavior bundles (steering data)
|   +-- DeepThinkingFlow/
|       |-- profile.json                  <- Bundle metadata, quality gates, file map
|       |-- system_prompt.txt             <- System prompt with hard_rules, depth_policy, etc.
|       |-- evals/
|       |   +-- reasoning_following.jsonl <- 20+ eval cases with trait + rubric rules
|       +-- training/
|           |-- sft_reasoning_vi.jsonl        <- 6+ original SFT examples (vi)
|           |-- harmony_sft_vi.jsonl          <- 49 harmony-format examples (vi)
|           |-- harmony_sft_vi.train.jsonl    <- 39 train split (fixed, seed=42)
|           +-- harmony_sft_vi.eval.jsonl     <- 10 eval split  (fixed, seed=42)
|
|-- original/                             <- Original upstream model snapshot
|   |-- config.json                       <- Architecture config (MoE, 24 layers, etc.)
|   |-- dtypes.json                       <- Weight dtype metadata (FP4, UE8, BF16)
|   +-- model.safetensors                 <- ~12.82 GiB weights (git-ignored)
|
|-- runtime/                              <- Transformers-ready model directory
|   +-- transformers/
|       +-- DeepThinkingFlow/
|           |-- bootstrap-manifest.json   <- Bootstrapped file manifest
|           |-- config.json               <- Transformers model config
|           |-- generation_config.json    <- Generation defaults
|           |-- chat_template.jinja       <- Chat template with channel routing
|           |-- tokenizer.json            <- ~26.6 MB tokenizer (201,088 vocab)
|           |-- tokenizer_config.json     <- Tokenizer settings
|           |-- special_tokens_map.json   <- Special token mapping
|           |-- dtypes.json               <- Symlink to original/dtypes.json
|           +-- model.safetensors         <- Symlink to original/model.safetensors
|
|-- scripts/                              <- All Python scripts (12 files)
|   |-- deepthinkingflow_cli.py               <- Unified CLI launcher
|   |-- deepthinkingflow_runtime.py           <- Shared runtime helpers
|   |-- chat_deepthinkingflow.py              <- Multi-turn terminal chat
|   |-- run_transformers_deepthinkingflow.py  <- One-shot generation (JSON output)
|   |-- render_transformers_deepthinkingflow_prompt.py  <- Prompt preview
|   |-- bootstrap_transformers_deepthinkingflow.py      <- Bootstrap from HF
|   |-- assemble_local_transformers_model_dir.py        <- Symlink weights overlay
|   |-- compose_behavior_request.py           <- Compose messages from bundle
|   |-- validate_behavior_bundle.py           <- Bundle health checker
|   |-- prepare_harmony_sft_dataset.py        <- Dataset dedupe + split
|   |-- train_transformers_deepthinkingflow_lora.py     <- LoRA/QLoRA trainer
|   +-- evaluate_reasoning_outputs.py         <- Heuristic eval scorer
|
|-- training/                             <- Training config templates
|   +-- DeepThinkingFlow-lora/
|       |-- config.example.json           <- LoRA config (bf16, r=8, alpha=16)
|       +-- config.qlora.example.json     <- QLoRA config (4-bit, paged_adamw_8bit)
|
|-- out/                                  <- Training outputs (git-ignored)
|   |-- DeepThinkingFlow-lora-reasoning-vi/
|   |   +-- run-manifest.json             <- Training run metadata
|   +-- DeepThinkingFlow-qlora-reasoning-vi/
|       +-- run-manifest.json             <- Training run metadata
|
|-- skills/                               <- Codex skill definitions
|   +-- DeepThinkingFlow/
|       |-- SKILL.md                      <- Skill instructions for AI assistants
|       |-- agents/
|       |   +-- openai.yaml              <- Agent interface config
|       +-- references/
|           |-- model-profile.md          <- MoE architecture facts
|           |-- reasoning-patterns.md     <- Reasoning behavior patterns
|           |-- prompt-templates.md       <- Reusable prompt scaffolds
|           |-- response-examples.md      <- Example answer templates
|           +-- runtime-and-training.md   <- Runtime & training guide
|
+-- tests/                                <- Unit tests
    +-- test_deepthinkingflow_smoke.py    <- 7 smoke tests (all passing)
```

---

## Prerequisites

### System Requirements

| Item | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.11+ |
| RAM | 16 GiB | 32 GiB+ |
| GPU VRAM | 16 GiB (QLoRA 4-bit) | 24 GiB+ (LoRA bf16) |
| Disk | 15 GiB (weights) | 30 GiB (weights + outputs) |

### Install Dependencies

**For inference (running the model):**
```bash
pip install -r requirements-transformers.txt
```

**For training (LoRA/QLoRA fine-tuning):**
```bash
pip install -r requirements-train-gpt-oss.txt

# If using QLoRA (4-bit quantization):
pip install bitsandbytes>=0.46.0
```

<details>
<summary>Dependency details</summary>

**Inference:**
| Package | Version |
|---|---|
| transformers | >=4.57.0, <5.0.0 |
| tokenizers | >=0.21.0, <1.0.0 |
| huggingface_hub | >=0.35.0, <1.0.0 |
| safetensors | >=0.6.0, <1.0.0 |
| jinja2 | >=3.1.0, <4.0.0 |

**Training (additional):**
| Package | Version |
|---|---|
| torch | >=2.7.0, <3.0.0 |
| accelerate | >=1.10.0, <2.0.0 |
| datasets | >=4.0.0, <5.0.0 |
| peft | >=0.17.0, <1.0.0 |

</details>

---

## Quick Start

### 1. Bootstrap the model directory from HuggingFace

```bash
# Download metadata (tokenizer, config, chat template) -- does NOT include weights
python scripts/deepthinkingflow_cli.py bootstrap

# Or include weights (~12.8 GiB):
python scripts/deepthinkingflow_cli.py bootstrap --include-weights
```

### 2. (Optional) Link local weights

If you already have `model.safetensors` in the `original/` directory:
```bash
python scripts/deepthinkingflow_cli.py assemble-model-dir
```

### 3. Interactive chat

```bash
python scripts/deepthinkingflow_cli.py chat
```

### 4. One-shot generation

```bash
python scripts/deepthinkingflow_cli.py run --user "Explain MoE architecture"
```

### 5. Validate the behavior bundle

```bash
python scripts/deepthinkingflow_cli.py validate-bundle behavior/DeepThinkingFlow
```

---

## CLI Reference

All scripts are accessed through the unified CLI launcher:

```bash
python scripts/deepthinkingflow_cli.py <command> [args]
```

| Command | Script | Description |
|---|---|---|
| `chat` | `chat_deepthinkingflow.py` | Interactive multi-turn chat with conversation history |
| `run` | `run_transformers_deepthinkingflow.py` | One-shot generation returning JSON |
| `render-prompt` | `render_transformers_deepthinkingflow_prompt.py` | Render the injected chat-template prompt |
| `compose-request` | `compose_behavior_request.py` | Compose messages from the behavior bundle |
| `validate-bundle` | `validate_behavior_bundle.py` | Validate bundle health |
| `bootstrap` | `bootstrap_transformers_deepthinkingflow.py` | Bootstrap model directory from HF |
| `assemble-model-dir` | `assemble_local_transformers_model_dir.py` | Symlink local weights into model dir |
| `prepare-sft` | `prepare_harmony_sft_dataset.py` | Deduplicate + split SFT dataset |
| `train-lora` | `train_transformers_deepthinkingflow_lora.py` | Train LoRA/QLoRA adapter |
| `eval` | `evaluate_reasoning_outputs.py` | Score outputs against trait + rubric |

### Chat Commands (inside a chat session)

```
/help                Show available commands
/status              Show current runtime settings
/clear               Clear history, keep system prompt
/history             Print the retained conversation
/analysis on|off     Toggle visible analysis output
/reasoning <level>   Switch reasoning effort: low, medium, high
/quit                Exit the chat session
```

---

## Workflows

### 1. Inference Workflow

> Use an existing model to generate answers.

```
                    +-------------------+
                    |   User starts     |
                    +---------+---------+
                              |
                              v
              +-------------------------------+
              |  1. Bootstrap Model Directory  |
              |  bootstrap --include-weights   |
              |  OR assemble-model-dir         |
              +---------------+---------------+
                              |
                              v
              +-------------------------------+
              |  2. Validate Behavior Bundle   |
              |  validate-bundle behavior/...  |
              +---------------+---------------+
                              |
                              v
                    +---------+---------+
                    |  Choose mode?     |
                    +---------+---------+
                    +---------+---------+
                    v                   v
          +--------------+    +------------------+
          |  One-shot     |    |  Multi-turn Chat  |
          |  run --user   |    |  chat              |
          |  "prompt"     |    |  (interactive)     |
          +------+-------+    +--------+----------+
                 |                     |
                 v                     v
          +--------------+    +------------------+
          |  JSON Output  |    |  Stream output    |
          |  {final_text} |    |  DeepThinkingFlow>|
          +--------------+    +------------------+
```

**Detailed steps:**

```bash
# Step 1: Prepare model
python scripts/deepthinkingflow_cli.py bootstrap
python scripts/deepthinkingflow_cli.py assemble-model-dir

# Step 2: Validate bundle
python scripts/deepthinkingflow_cli.py validate-bundle behavior/DeepThinkingFlow

# Step 3a: One-shot
python scripts/deepthinkingflow_cli.py run \
  --user "Analyze this prompt" \
  --reasoning-effort high \
  --include-analysis

# Step 3b: Chat
python scripts/deepthinkingflow_cli.py chat \
  --reasoning-effort high \
  --show-analysis \
  --max-history-turns 6
```

---

### 2. Training Workflow

> Train a LoRA/QLoRA adapter to improve model behavior.

```
 +-----------------------------------------------------------------+
 |                   TRAINING PIPELINE                              |
 +-----------------------------------------------------------------+
 |                                                                  |
 |  (1) Prepare Dataset                                             |
 |  +------------------+     +-----------------------------+        |
 |  | harmony_sft_vi   |---->| prepare_harmony_sft_dataset |        |
 |  | .jsonl (49 ex)   |     | --eval-ratio 0.2 --seed 42  |        |
 |  +------------------+     +----------+------------------+        |
 |                                      |                           |
 |                           +----------+----------+                |
 |                           v                     v                |
 |                    train.jsonl (39)       eval.jsonl (10)         |
 |                                                                  |
 |  (2) Dry Run (Validate)                                          |
 |  +------------------------------------------------------+       |
 |  | train-lora --config config.example.json --dry-run     |       |
 |  | -> Validates config, loads tokenizer, preprocesses    |       |
 |  | -> Outputs summary JSON + run-manifest.json           |       |
 |  +--------------------------------------+---------------+       |
 |                                         |                        |
 |  (3) Train                              v                        |
 |  +------------------------------------------------------+       |
 |  | train-lora --config config.example.json               |       |
 |  | -> Loads base model (bf16 or 4-bit)                   |       |
 |  | -> Applies LoraConfig (r=8, alpha=16, dropout=0.05)   |       |
 |  | -> HF Trainer with EarlyStopping                      |       |
 |  | -> Saves adapter to out/ directory                    |       |
 |  +--------------------------------------+---------------+       |
 |                                         |                        |
 |  (4) Evaluate                           v                        |
 |  +------------------------------------------------------+       |
 |  | eval --eval-cases evals/reasoning_following.jsonl     |       |
 |  |      --predictions predictions.jsonl                  |       |
 |  | -> Scores: trait_pass_rate + rubric_pass_rate          |       |
 |  +--------------------------------------+---------------+       |
 |                                         |                        |
 |  (5) (Optional) Merge                   v                        |
 |  +------------------------------------------------------+       |
 |  | Set "merge_after_train": true in config               |       |
 |  | -> PeftModel.merge_and_unload()                       |       |
 |  | -> Saves merged model to out/*-merged/                |       |
 |  +------------------------------------------------------+       |
 |                                                                  |
 +-----------------------------------------------------------------+
```

**Detailed steps:**

```bash
# Step 1: Prepare dataset (if fixed splits do not exist yet)
python scripts/deepthinkingflow_cli.py prepare-sft \
  --input behavior/DeepThinkingFlow/training/harmony_sft_vi.jsonl \
  --train-out behavior/DeepThinkingFlow/training/harmony_sft_vi.train.jsonl \
  --eval-out behavior/DeepThinkingFlow/training/harmony_sft_vi.eval.jsonl \
  --eval-ratio 0.2 --seed 42

# Step 2: Dry run
python scripts/deepthinkingflow_cli.py train-lora \
  --config training/DeepThinkingFlow-lora/config.example.json \
  --dry-run

# Step 3: Train (LoRA)
python scripts/deepthinkingflow_cli.py train-lora \
  --config training/DeepThinkingFlow-lora/config.example.json

# Or Train (QLoRA -- saves VRAM)
python scripts/deepthinkingflow_cli.py train-lora \
  --config training/DeepThinkingFlow-lora/config.qlora.example.json

# Step 4: Evaluate
python scripts/deepthinkingflow_cli.py eval \
  --eval-cases behavior/DeepThinkingFlow/evals/reasoning_following.jsonl \
  --predictions your_predictions.jsonl
```

---

### 3. Evaluation Workflow

> Score output quality along two dimensions: **traits** and **rubrics**.

```
                  +--------------------+
                  |  eval_cases.jsonl  |  <- Each case has:
                  |                    |     id, user, expected_traits,
                  |                    |     required_keywords, rubric rules
                  +----------+---------+
                             |
                             v
              +--------------------------+
              |  predictions.jsonl       |  <- Each row has:
              |  (from model run)        |     id, final_text, analysis_text
              +-------------+------------+
                            |
                            v
              +--------------------------+
              |  evaluate_reasoning_     |
              |  outputs.py              |
              |                          |
              |  Scoring:                |
              |  +-- Trait scoring       |  <- 18 trait types
              |  |   (keyword match,     |     (simple_definition,
              |  |    length check,      |      concise_reasoning,
              |  |    structure check)   |      phased_plan, ...)
              |  +-- Rubric scoring      |  <- Rule-based checks
              |      (required_keywords, |     (keyword groups,
              |       forbidden_keywords,|      max_chars,
              |       must_start_with,   |      min_numbered_steps)
              |       max_chars, ...)    |
              +-------------+------------+
                            |
                            v
              +--------------------------+
              |  Output Summary JSON     |
              |  {                       |
              |    trait_pass_rate: 0.85, |
              |    rubric_pass_rate: 0.9, |
              |    results: [...]        |
              |  }                       |
              +--------------------------+
```

**Supported Traits:**

| Trait | Check |
|---|---|
| `simple_definition` | First line under 180 characters |
| `short_analysis` | Analysis under 400 characters |
| `one_concrete_example` | Contains "example" or Vietnamese equivalent |
| `concise_reasoning` | Full output under 1,400 characters |
| `likely_causes_first` | Lists probable causes before fixes |
| `ordered_checks` | Contains numbered check steps |
| `probable_fix` | Contains a concrete fix or solution |
| `findings_first` | First line leads with findings |
| `security_risk_called_out` | Mentions security or risk |
| `recommendation_first` | First line leads with a recommendation |
| `3_to_5_criteria` | At least 3 comparison criteria present |
| `one_tradeoff` | Mentions at least one tradeoff |
| `phased_plan` | Contains "phase 1/2" or equivalent |
| `validation_step` | Includes a validation or benchmark step |
| `rollback_step` | Includes a rollback or fallback plan |
| `main_risk` | Identifies the main risk |
| `brief_summary` | Full output under 1,600 characters |
| `scenario_example` | Contains a scenario-based example |

---

### 4. Full Pipeline (End-to-End)

```
+---------+   +------------+   +-----------+   +----------+   +-----------+   +------------+
|  Write  |-->|  Validate  |-->|  Prepare  |-->|  Train   |-->|  Generate |-->|  Evaluate  |
|  Data   |   |  Bundle    |   |  Dataset  |   |  LoRA    |   |  Predict  |   |  & Compare |
+---------+   +------------+   +-----------+   +----------+   +-----------+   +------------+
     |              |               |               |               |               |
 harmony_      validate-       prepare-sft      train-lora       run --user       eval
 sft_vi         bundle                                            "..."        --eval-cases
 .jsonl
```

---

## Behavior Bundle System

A behavior bundle is the central mechanism for **steering model behavior without modifying weights**.

### Bundle Structure

```
behavior/DeepThinkingFlow/
|-- profile.json          <- Metadata + quality gates + file references
|-- system_prompt.txt     <- System prompt (injected into every request)
|-- evals/
|   +-- reasoning_following.jsonl  <- Eval cases
+-- training/
    |-- sft_reasoning_vi.jsonl          <- Raw SFT seed data
    |-- harmony_sft_vi.jsonl            <- Full harmony dataset
    |-- harmony_sft_vi.train.jsonl      <- Fixed train split
    +-- harmony_sft_vi.eval.jsonl       <- Fixed eval split
```

### Profile Guarantees

```json
{
  "guarantees": {
    "does_not_modify_weights": true,
    "does_not_claim_model_retraining": true,
    "requires_runtime_integration": true
  }
}
```

### System Prompt Structure

The system prompt uses tagged blocks:

| Block | Purpose |
|---|---|
| `<identity>` | Assistant identity declaration |
| `<hard_rules>` | Mandatory rules (language, transparency, verification) |
| `<task_classifier>` | Classifies tasks: explain, debug, review, compare, plan, estimate |
| `<depth_policy>` | Three levels: Quick, Standard, Deep |
| `<output_policy>` | Output format per task type |
| `<local_model_guidance>` | Optimization guidance for local models |
| `<quality_bar>` | Quality standards |

### Quality Gates

The bundle is automatically validated via `validate-bundle`:

| Gate | Value |
|---|---|
| `min_sft_examples` | >= 6 |
| `min_harmony_sft_examples` | >= 45 |
| `min_eval_cases` | >= 20 |
| `require_unique_eval_ids` | `true` |
| `require_unique_harmony_examples` | `true` |

---

## Model Profile

| Property | Value |
|---|---|
| **Upstream** | `openai/gpt-oss-20b` |
| **Architecture** | Transformer + Mixture-of-Experts (MoE) |
| **Layers** | 24 |
| **Hidden size** | 2,880 |
| **Vocab size** | 201,088 |
| **Attention** | 64 query heads, 8 KV heads, dim=64 |
| **Experts** | 32 per layer, 4 active per token |
| **Context** | 4,096 tokens initial, sliding window 128 |
| **Total params (est.)** | ~21.5B (when expanding packed FP4) |
| **Active params/token (est.)** | ~4.19B (4/32 experts) |
| **Weight format** | BF16 (attention/embedding) + Packed FP4 + UE8 scales (MoE) |
| **File size** | ~12.82 GiB |

### Special Tokens and Channel System

The model uses a channel system to separate reasoning from output:

```
<|start|>assistant<|channel|>analysis<|message|>...<|end|>
<|start|>assistant<|channel|>final<|message|>...<|return|>
```

- **`analysis`** -- Visible reasoning (hidden by default; enable via `--show-analysis` or `/analysis on`)
- **`final`** -- The final answer shown to the user

---

## Training Configuration

### LoRA Config (`config.example.json`)

| Parameter | Value | Description |
|---|---|---|
| `lora_r` | 8 | Rank of LoRA matrices |
| `lora_alpha` | 16 | Scaling factor |
| `lora_dropout` | 0.05 | Dropout rate |
| `target_modules` | `[q_proj, k_proj, v_proj, o_proj]` | Attention projection layers |
| `bf16` | `true` | BFloat16 precision |
| `learning_rate` | 0.0002 | Peak learning rate |
| `lr_scheduler_type` | `cosine` | Cosine decay scheduler |
| `gradient_checkpointing` | `true` | Saves VRAM |
| `gradient_accumulation_steps` | 8 | Effective batch = 1 x 8 = 8 |
| `max_seq_length` | 4,096 | Maximum sequence length |
| `early_stopping_patience` | 3 | Stop if eval_loss does not improve for 3 consecutive evals |
| `optim` | `adamw_torch` | Optimizer |

### QLoRA Config (`config.qlora.example.json`)

Same as LoRA, with these additions:

| Parameter | Value | Description |
|---|---|---|
| `use_qlora` | `true` | Enables QLoRA mode |
| `load_in_4bit` | `true` | Loads model in 4-bit (NF4) |
| `optim` | `paged_adamw_8bit` | Memory-efficient optimizer |

> **Note:** QLoRA requires the `bitsandbytes` package.

---

## Testing

### Smoke Tests (7/7)

```bash
python -m pytest tests/test_deepthinkingflow_smoke.py -v
```

| Test | Description |
|---|---|
| `RuntimeHelpersTest::test_extracts_analysis_and_final_text` | Verifies channel token extraction |
| `CliSmokeTest::test_help_dispatches_to_subcommand_help` | CLI `help` routing |
| `CliSmokeTest::test_unknown_command_returns_error` | CLI unknown command returns exit code 2 |
| `CliSmokeTest::test_dispatch_builds_expected_subprocess_call` | CLI subprocess argument construction |
| `RenderPromptSmokeTest::test_render_prompt_main_with_fake_tokenizer` | Prompt rendering pipeline |
| `RunSmokeTest::test_run_main_returns_expected_json_without_loading_real_model` | One-shot generation flow |
| `ChatSmokeTest::test_chat_main_handles_commands_and_response_flow` | Full chat lifecycle |

> Tests use mocks and run without a GPU or real model weights.

---

## Codex Skill Integration

The `skills/DeepThinkingFlow/` directory provides guidance for AI coding assistants (Codex, etc.):

```
skills/DeepThinkingFlow/
|-- SKILL.md                        <- Main instructions
|-- agents/openai.yaml              <- Agent interface config
+-- references/
    |-- model-profile.md            <- Architecture & prompting implications
    |-- reasoning-patterns.md       <- Reasoning behavior patterns
    |-- prompt-templates.md         <- Reusable prompt scaffolds
    |-- response-examples.md        <- Answer templates
    +-- runtime-and-training.md     <- Runtime & training integration guide
```

### Skill Workflow

```
1. Classify task       -> explain | debug | review | compare | plan | estimate
2. Extract constraints -> language, depth, format, risk, evidence
3. Choose depth        -> Quick | Standard | Deep
4. Select prompt scaffold (if rewriting)
5. Select answer pattern
6. Final check for missing caveats & unsupported claims
```

### Output Contract

```
Goal:        <one-sentence restatement>
Assumptions: <only if needed>
Analysis:    <short visible reasoning>
Answer:      <direct answer or recommendation>
Examples:    <1-3 concrete examples>
Checks:      <verification, caveat, or next step>
```

---

## Dataset Statistics

| Dataset | Count | Description |
|---|---|---|
| `sft_reasoning_vi.jsonl` | 6+ examples | Original SFT seed (Vietnamese) |
| `harmony_sft_vi.jsonl` | 49 examples | Full harmony-format dataset |
| `harmony_sft_vi.train.jsonl` | 39 examples | Fixed train split (seed=42) |
| `harmony_sft_vi.eval.jsonl` | 10 examples | Fixed eval split (seed=42) |
| `reasoning_following.jsonl` | 20+ cases | Eval cases with traits + rubric |

---

## Design Principles

1. **Transparency** -- No claims of hidden chain-of-thought or secret reasoning
2. **Separation of Concerns** -- Behavior bundle is decoupled from model weights
3. **Reproducibility** -- Fixed train/eval splits, deterministic seeds
4. **Safety** -- Low-memory warnings, config validation, dry-run mode
5. **Bilingual** -- Vietnamese-first, English-compatible
6. **Modularity** -- Each script does one thing; the CLI orchestrates everything

---

## License

This project is released under the [GNU General Public License v3.0](LICENSE).

---

<p align="center">
  <strong>DeepThinkingFlow-AI</strong> -- by <a href="https://github.com/danggiaminh">Dang Gia Minh</a><br/>
  <sub>Runtime steering | Bilingual reasoning | Adapter-based fine-tuning | Open source</sub>
</p>

