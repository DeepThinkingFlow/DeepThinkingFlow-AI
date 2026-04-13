from __future__ import annotations

import contextlib
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import chat_deepthinkingflow as chat
import deepthinkingflow_cli as cli
import deepthinkingflow_runtime as runtime
import render_transformers_deepthinkingflow_prompt as render_prompt
import run_transformers_deepthinkingflow as run_script


MODEL_DIR = str((ROOT_DIR / "runtime" / "transformers" / "DeepThinkingFlow").resolve())
BUNDLE_DIR = str((ROOT_DIR / "behavior" / "DeepThinkingFlow").resolve())


class RuntimeHelpersTest(unittest.TestCase):
    def test_extracts_analysis_and_final_text(self) -> None:
        decoded = (
            "<|channel|>analysis<|message|>phan tich ngan<|end|>"
            "<|channel|>final<|message|>tra loi cuoi<|return|>"
        )

        self.assertEqual(runtime.extract_analysis_text(decoded), "phan tich ngan")
        self.assertEqual(runtime.extract_final_text(decoded), "tra loi cuoi")


class CliSmokeTest(unittest.TestCase):
    def test_help_dispatches_to_subcommand_help(self) -> None:
        with mock.patch.object(cli, "dispatch", return_value=0) as dispatch:
            with mock.patch.object(sys, "argv", ["deepthinkingflow_cli.py", "help", "chat"]):
                result = cli.main()

        self.assertEqual(result, 0)
        dispatch.assert_called_once_with("chat", ["--help"])

    def test_unknown_command_returns_error(self) -> None:
        stderr = io.StringIO()
        stdout = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
            with mock.patch.object(sys, "argv", ["deepthinkingflow_cli.py", "missing-command"]):
                result = cli.main()

        self.assertEqual(result, 2)
        self.assertIn("Unknown command", stderr.getvalue())
        self.assertIn("DeepThinkingFlow CLI", stdout.getvalue())

    def test_dispatch_builds_expected_subprocess_call(self) -> None:
        completed = types.SimpleNamespace(returncode=7)
        with mock.patch("deepthinkingflow_cli.subprocess.run", return_value=completed) as run_mock:
            result = cli.dispatch("chat", ["--help"])

        self.assertEqual(result, 7)
        run_mock.assert_called_once()
        args, kwargs = run_mock.call_args
        self.assertEqual(args[0][0], sys.executable)
        self.assertTrue(args[0][1].endswith("chat_deepthinkingflow.py"))
        self.assertEqual(args[0][2:], ["--help"])
        self.assertEqual(kwargs["cwd"], str(ROOT_DIR))


class RenderPromptSmokeTest(unittest.TestCase):
    def test_render_prompt_main_with_fake_tokenizer(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt, reasoning_effort):
                self.last_messages = messages
                self.last_reasoning_effort = reasoning_effort
                return f"RENDERED::{reasoning_effort}::{messages[-1]['content']}"

        fake_tokenizer = FakeTokenizer()
        fake_transformers = types.ModuleType("transformers")

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(_model_dir):
                return fake_tokenizer

        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        stdout = io.StringIO()

        argv = [
            "render_transformers_deepthinkingflow_prompt.py",
            "--model-dir",
            MODEL_DIR,
            "--bundle",
            BUNDLE_DIR,
            "--user",
            "Xin chao",
            "--json",
        ]

        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(stdout):
                    result = render_prompt.main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["messages"][-1]["content"], "Xin chao")
        self.assertEqual(payload["rendered_prompt"], "RENDERED::high::Xin chao")


class RunSmokeTest(unittest.TestCase):
    def test_run_main_returns_expected_json_without_loading_real_model(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "run_transformers_deepthinkingflow.py",
            "--model-dir",
            MODEL_DIR,
            "--bundle",
            BUNDLE_DIR,
            "--user",
            "Kiem tra one-shot",
            "--include-analysis",
            "--include-raw-completion",
        ]

        with mock.patch.object(run_script, "build_low_memory_warning_payload", return_value=None):
            with mock.patch.object(run_script, "load_model_and_tokenizer", return_value=("TOKENIZER", "MODEL")):
                with mock.patch.object(run_script, "render_prompt", return_value="PROMPT") as render_mock:
                    with mock.patch.object(
                        run_script,
                        "generate_response",
                        return_value={
                            "analysis_text": "phan tich test",
                            "final_text": "ket qua test",
                            "decoded_completion": "RAW",
                        },
                    ) as generate_mock:
                        with mock.patch.object(sys, "argv", argv):
                            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                                result = run_script.main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["final_text"], "ket qua test")
        self.assertEqual(payload["analysis_text"], "phan tich test")
        self.assertEqual(payload["decoded_completion"], "RAW")
        render_mock.assert_called_once()
        generate_mock.assert_called_once()
        self.assertEqual(stderr.getvalue(), "")


class ChatSmokeTest(unittest.TestCase):
    def test_chat_main_handles_commands_and_response_flow(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        captured_messages: list[list[dict[str, str]]] = []
        argv = [
            "chat_deepthinkingflow.py",
            "--model-dir",
            MODEL_DIR,
            "--bundle",
            BUNDLE_DIR,
            "--max-history-turns",
            "2",
        ]

        def fake_generate_response(*_args, **kwargs):
            captured_messages.append([dict(message) for message in kwargs["messages"]])
            return {
                "analysis_text": "phan tich test",
                "final_text": "Xin chao tu test",
                "decoded_completion": "RAW",
            }

        with mock.patch.object(chat, "build_low_memory_warning_payload", return_value=None):
            with mock.patch.object(chat, "load_model_and_tokenizer", return_value=("TOKENIZER", "MODEL")):
                with mock.patch.object(
                    chat,
                    "generate_response",
                    side_effect=fake_generate_response,
                ) as generate_mock:
                    with mock.patch(
                        "builtins.input",
                        side_effect=[
                            "/status",
                            "/analysis on",
                            "/reasoning medium",
                            "Xin chao",
                            "/history",
                            "/clear",
                            "/quit",
                        ],
                    ):
                        with mock.patch.object(sys, "argv", argv):
                            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                                result = chat.main()

        output = stdout.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("DeepThinkingFlow terminal chat is ready.", output)
        self.assertIn("Status:", output)
        self.assertIn("Visible analysis is now on.", output)
        self.assertIn("Reasoning effort switched to medium.", output)
        self.assertIn("Analysis>", output)
        self.assertIn("phan tich test", output)
        self.assertIn("DeepThinkingFlow>", output)
        self.assertIn("Xin chao tu test", output)
        self.assertIn("[1] user: Xin chao", output)
        self.assertIn("[2] assistant: Xin chao tu test", output)
        self.assertIn("Conversation history cleared.", output)
        self.assertIn("DeepThinkingFlow chat closed.", output)
        self.assertEqual(stderr.getvalue(), "")

        self.assertEqual(generate_mock.call_count, 1)
        kwargs = generate_mock.call_args.kwargs
        self.assertEqual(kwargs["reasoning_effort"], "medium")
        self.assertEqual(captured_messages[0][-1]["content"], "Xin chao")


if __name__ == "__main__":
    unittest.main()
