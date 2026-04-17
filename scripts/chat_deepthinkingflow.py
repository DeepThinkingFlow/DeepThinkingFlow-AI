#!/usr/bin/env python3
"""Interactive terminal chat for DeepThinkingFlow with local behavior injection."""

from __future__ import annotations

import argparse
import sys

from deepthinkingflow_runtime import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_MODEL_DIR,
    build_low_memory_warning_payload,
    generate_response,
    load_model_and_tokenizer,
    load_system_prompt,
    resolve_bundle_dir,
    resolve_model_ref,
)

CHAT_HELP = """Commands:
  /help                Show this help.
  /status              Show current runtime settings.
  /clear               Clear chat history and keep only the system prompt.
  /history             Print the current retained conversation.
  /analysis on|off     Toggle sanitized visible analysis output.
  /reasoning <level>   Switch reasoning effort: low, medium, or high.
  /quit                Exit the chat session.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with DeepThinkingFlow in a multi-turn terminal session."
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Transformers-ready local model directory or HF model id.",
    )
    parser.add_argument(
        "--bundle",
        default=DEFAULT_BUNDLE_DIR,
        help="Behavior bundle directory.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="high",
        help="Initial reasoning effort passed into the chat template.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help="Maximum number of generated tokens per reply.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map argument.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Transformers torch_dtype argument.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional attention implementation override.",
    )
    parser.add_argument(
        "--show-analysis",
        action="store_true",
        help="Show sanitized extracted analysis text for debugging.",
    )
    parser.add_argument(
        "--reasoning-in-system",
        action="store_true",
        help="Also append 'Reasoning: <level>' to the system prompt text.",
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=6,
        help="Retain only the latest N user/assistant turns. Use 0 to disable trimming.",
    )
    return parser.parse_args()


def rebuild_system_message(
    messages: list[dict[str, str]],
    bundle_dir,
    reasoning_effort: str,
    reasoning_in_system: bool,
) -> None:
    messages[0] = {
        "role": "system",
        "content": load_system_prompt(
            bundle_dir,
            reasoning_effort=reasoning_effort,
            reasoning_in_system=reasoning_in_system,
        ),
    }


def trim_messages(messages: list[dict[str, str]], max_history_turns: int) -> list[dict[str, str]]:
    if max_history_turns <= 0 or len(messages) <= 1:
        return messages
    retained = messages[1:]
    max_messages = max_history_turns * 2
    if len(retained) <= max_messages:
        return messages
    return [messages[0], *retained[-max_messages:]]


def print_status(
    *,
    model_ref: str,
    reasoning_effort: str,
    show_analysis: bool,
    max_history_turns: int,
    messages: list[dict[str, str]],
) -> None:
    history_messages = max(0, len(messages) - 1)
    print("Status:")
    print(f"  model: {model_ref}")
    print(f"  reasoning_effort: {reasoning_effort}")
    print(f"  show_analysis: {'on' if show_analysis else 'off'}")
    print(f"  max_history_turns: {max_history_turns}")
    print(f"  retained_messages: {history_messages}")


def print_history(messages: list[dict[str, str]]) -> None:
    if len(messages) <= 1:
        print("History is empty.")
        return
    print("Retained conversation:")
    for index, message in enumerate(messages[1:], start=1):
        role = message["role"]
        content = message["content"].strip() or "(empty)"
        print(f"[{index}] {role}: {content}")


def print_banner(model_ref: str) -> None:
    print("DeepThinkingFlow terminal chat is ready.")
    print(f"Model: {model_ref}")
    print("Type /help for commands, /quit to exit.")


def main() -> int:
    args = parse_args()
    if args.max_history_turns < 0:
        raise SystemExit("--max-history-turns must be >= 0")

    bundle_dir = resolve_bundle_dir(args.bundle)
    model_ref, model_path = resolve_model_ref(args.model_dir)
    warning = build_low_memory_warning_payload(model_path)
    if warning:
        print(
            (
                "[warn] Low-memory host detected for Transformers inference. "
                f"RAM={warning['system_ram_gib']} GiB, weights={warning['local_weight_gib']} GiB."
            ),
            file=sys.stderr,
        )

    tokenizer, model = load_model_and_tokenizer(
        model_ref,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    reasoning_effort = args.reasoning_effort
    show_analysis = args.show_analysis
    messages = [{"role": "system", "content": ""}]
    rebuild_system_message(messages, bundle_dir, reasoning_effort, args.reasoning_in_system)

    print_banner(model_ref)

    while True:
        try:
            user_text = input("You> ").strip()
        except EOFError:
            print("\nDeepThinkingFlow chat closed.")
            return 0
        except KeyboardInterrupt:
            print("\nUse /quit to exit, or press Ctrl-D.")
            continue

        if not user_text:
            continue

        if user_text.startswith("/"):
            command, _, value = user_text.partition(" ")
            value = value.strip()

            if command in {"/quit", "/exit"}:
                print("DeepThinkingFlow chat closed.")
                return 0
            if command == "/help":
                print(CHAT_HELP.rstrip())
                continue
            if command == "/clear":
                messages = [{"role": "system", "content": ""}]
                rebuild_system_message(messages, bundle_dir, reasoning_effort, args.reasoning_in_system)
                print("Conversation history cleared.")
                continue
            if command == "/history":
                print_history(messages)
                continue
            if command == "/status":
                print_status(
                    model_ref=model_ref,
                    reasoning_effort=reasoning_effort,
                    show_analysis=show_analysis,
                    max_history_turns=args.max_history_turns,
                    messages=messages,
                )
                continue
            if command == "/analysis":
                if value not in {"on", "off"}:
                    print("Usage: /analysis on|off")
                    continue
                show_analysis = value == "on"
                print(f"Visible analysis is now {'on' if show_analysis else 'off'}.")
                continue
            if command == "/reasoning":
                if value not in {"low", "medium", "high"}:
                    print("Usage: /reasoning low|medium|high")
                    continue
                reasoning_effort = value
                rebuild_system_message(messages, bundle_dir, reasoning_effort, args.reasoning_in_system)
                print(f"Reasoning effort switched to {reasoning_effort}.")
                continue

            print("Unknown command. Type /help for the supported commands.")
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            response = generate_response(
                model,
                tokenizer,
                messages=messages,
                reasoning_effort=reasoning_effort,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except KeyboardInterrupt:
            messages.pop()
            print("\nGeneration interrupted.")
            continue
        except Exception as exc:
            messages.pop()
            print(f"[error] Generation failed: {exc}", file=sys.stderr)
            continue

        final_text = response["final_text"].strip() or "[No final answer emitted.]"
        messages.append({"role": "assistant", "content": final_text})
        messages = trim_messages(messages, args.max_history_turns)

        if show_analysis and response["analysis_text"]:
            print("Analysis>")
            print(response["analysis_text"])
        print("DeepThinkingFlow>")
        print(final_text)


if __name__ == "__main__":
    raise SystemExit(main())
