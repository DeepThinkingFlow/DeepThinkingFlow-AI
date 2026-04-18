from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
import struct
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import chat_deepthinkingflow as chat
import compile_behavior_bundle as compile_bundle
import deepthinkingflow_cli as cli
import deepthinkingflow_env as dtf_env
import export_external_runtime_assets as export_runtime
import build_external_training_bundle as build_external_bundle
import export_prepared_chat_jsonl as export_chat_jsonl
import preflight_deepthinkingflow_project as preflight_project
import preflight_deepthinkingflow_training as preflight_train
import build_release_manifest as release_manifest
import deepthinkingflow_runtime as runtime
import verify_deepthinkingflow_project as verify_project
import evaluate_reasoning_outputs as evaluator
import inspect_safetensors_model as inspector
import prepare_deepthinkingflow_training_assets as asset_builder
import doctor_deepthinkingflow as doctor_script
import report_deepthinkingflow_artifacts as artifact_report
import run_tiny_smoke_release_lane as tiny_release_lane
import render_transformers_deepthinkingflow_prompt as render_prompt
import run_transformers_deepthinkingflow as run_script
import deepthinkingflow_system_check as system_check
import train_transformers_deepthinkingflow_lora as train_script
import train_deepthinkingflow_staged as staged_train
import validate_behavior_bundle as bundle_validator


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

    def test_sanitizes_visible_analysis_and_strips_channel_lines(self) -> None:
        decoded = (
            "<|channel|>analysis<|message|>analysis\n"
            "Kiem tra gia dinh chinh.\n"
            "return\n"
            "<|end|><|channel|>final<|message|>ket qua<|return|>"
        )

        self.assertEqual(runtime.extract_analysis_text(decoded), "Kiem tra gia dinh chinh.")

    def test_truncates_long_visible_analysis(self) -> None:
        long_line = "a" * 900
        decoded = f"<|channel|>analysis<|message|>{long_line}<|end|><|channel|>final<|message|>ket qua<|return|>"

        analysis = runtime.extract_analysis_text(decoded)

        self.assertTrue(analysis.endswith("..."))
        self.assertLessEqual(len(analysis), runtime.VISIBLE_ANALYSIS_MAX_CHARS + 3)


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

    def test_inspect_weights_command_is_registered(self) -> None:
        self.assertIn("inspect-weights", cli.COMMANDS)

    def test_prepare_training_assets_command_is_registered(self) -> None:
        self.assertIn("prepare-training-assets", cli.COMMANDS)

    def test_generate_skill_compliance_command_is_registered(self) -> None:
        self.assertIn("generate-skill-compliance", cli.COMMANDS)

    def test_report_artifacts_command_is_registered(self) -> None:
        self.assertIn("report-artifacts", cli.COMMANDS)

    def test_bootstrap_training_env_command_is_registered(self) -> None:
        self.assertIn("bootstrap-training-env", cli.COMMANDS)

    def test_preflight_train_command_is_registered(self) -> None:
        self.assertIn("preflight-train", cli.COMMANDS)

    def test_compile_bundle_command_is_registered(self) -> None:
        self.assertIn("compile-bundle", cli.COMMANDS)

    def test_system_check_command_is_registered(self) -> None:
        self.assertIn("system-check", cli.COMMANDS)

    def test_export_runtime_command_is_registered(self) -> None:
        self.assertIn("export-runtime", cli.COMMANDS)

    def test_preflight_all_command_is_registered(self) -> None:
        self.assertIn("preflight-all", cli.COMMANDS)

    def test_doctor_command_is_registered(self) -> None:
        self.assertIn("doctor", cli.COMMANDS)

    def test_verify_command_is_registered(self) -> None:
        self.assertIn("verify", cli.COMMANDS)

    def test_tiny_smoke_release_command_is_registered(self) -> None:
        self.assertIn("tiny-smoke-release", cli.COMMANDS)

    def test_prepare_datasets_command_is_registered(self) -> None:
        self.assertIn("prepare-datasets", cli.COMMANDS)

    def test_export_chat_jsonl_command_is_registered(self) -> None:
        self.assertIn("export-chat-jsonl", cli.COMMANDS)

    def test_build_external_train_bundle_command_is_registered(self) -> None:
        self.assertIn("build-external-train-bundle", cli.COMMANDS)

    def test_release_manifest_command_is_registered(self) -> None:
        self.assertIn("release-manifest", cli.COMMANDS)


class SystemCheckSmokeTest(unittest.TestCase):
    def test_format_warning_lines_is_soft_gate(self) -> None:
        report = {
            "profile": "inference",
            "warnings": [
                "System RAM is below the suggested minimum.",
                "No supported NVIDIA GPU was detected.",
            ],
        }

        lines = system_check.format_warning_lines(report)

        self.assertIn("soft gate", "\n".join(lines))
        self.assertGreaterEqual(len(lines), 3)

    def test_detect_external_runtime_status_has_expected_keys(self) -> None:
        status = dtf_env.detect_external_runtime_status()
        self.assertEqual(set(status), {"ollama", "claude", "claude_code"})


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


class ExternalRuntimeExportSmokeTest(unittest.TestCase):
    def test_export_ollama_assets_writes_runtime_only_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            argv = [
                "export_external_runtime_assets.py",
                "--bundle",
                BUNDLE_DIR,
                "--target",
                "ollama",
                "--ollama-model",
                "llama3.1:8b",
                "--output-dir",
                tmpdir,
            ]

            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(stdout):
                    result = export_runtime.main()

            payload = json.loads(stdout.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(payload["schema_version"], "dtf-external-runtime-export/v2")
            self.assertEqual(payload["target"], "ollama")
            self.assertEqual(payload["claim_level"], "runtime-only")
            self.assertIn("Modelfile", payload["created_files"])
            self.assertEqual(len(payload["file_reports"]), len(payload["created_files"]))
            self.assertTrue((Path(tmpdir) / "system_prompt.txt").is_file())
            self.assertTrue((Path(tmpdir) / "request.json").is_file())
            self.assertTrue((Path(tmpdir) / "Modelfile").is_file())
            self.assertIn("FROM llama3.1:8b", (Path(tmpdir) / "Modelfile").read_text(encoding="utf-8"))

    def test_export_runtime_can_fail_fast_when_host_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "export_external_runtime_assets.py",
                "--bundle",
                BUNDLE_DIR,
                "--target",
                "ollama",
                "--fail-if-host-missing",
                "--output-dir",
                tmpdir,
            ]
            with mock.patch.object(export_runtime, "detect_external_runtime_status", return_value={"ollama": False, "claude": False, "claude_code": False}):
                with mock.patch.object(sys, "argv", argv):
                    with self.assertRaises(SystemExit) as ctx:
                        export_runtime.main()
        self.assertIn("target=ollama", str(ctx.exception))


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

        fake_report = {"profile": "inference", "warnings": ["System RAM is below the suggested minimum."]}
        with mock.patch.object(run_script, "build_system_report", return_value=fake_report):
            with mock.patch.object(
                run_script,
                "format_system_warning_lines",
                return_value=["[warn] test system warning"],
            ):
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
        self.assertIn("[warn] test system warning", stderr.getvalue())


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

        with mock.patch.object(chat, "build_system_report", return_value={"profile": "inference", "warnings": []}):
            with mock.patch.object(chat, "format_system_warning_lines", return_value=[]):
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


class BundleValidationSmokeTest(unittest.TestCase):
    def test_validate_bundle_reports_skill_compliance_examples(self) -> None:
        summary = bundle_validator.validate_bundle(Path(BUNDLE_DIR))

        self.assertGreaterEqual(summary["skill_compliance_examples"], 48)
        self.assertEqual(summary["skill_compliance_eval_cases"], 24)
        self.assertEqual(
            summary["prepared_train_dataset"],
            "training/harmony_sft_plus_skill_compliance_vi.train.jsonl",
        )
        self.assertEqual(
            summary["prepared_eval_dataset"],
            "training/harmony_sft_plus_skill_compliance_vi.eval.jsonl",
        )
        self.assertEqual(
            summary["skill_compliance_categories"],
            {
                "deep-style-without-fake-internals": 12,
                "reject-false-weight-claim": 12,
                "runtime-vs-learned": 12,
                "short-analysis-no-cot": 12,
            },
        )


class EvaluatorSmokeTest(unittest.TestCase):
    def test_scores_new_skill_compliance_traits(self) -> None:
        final_text = (
            "Chưa. Đây mới là runtime-only vì prompt và wrapper không tự sửa weights trong model.safetensors. "
            "Muốn claim ở mức weights thì cần train LoRA hoặc QLoRA, có adapter artifact, eval, rồi merge hoặc checkpoint mới."
        )
        analysis_text = "Tom tat ngan, khong lo marker noi bo."

        self.assertTrue(evaluator.score_trait("explicit_runtime_only_boundary", final_text, analysis_text))
        self.assertTrue(evaluator.score_trait("explicit_training_boundary", final_text, analysis_text))
        self.assertTrue(evaluator.score_trait("explicit_no_weight_claim", final_text, analysis_text))
        self.assertTrue(evaluator.score_trait("analysis_sanitized", final_text, analysis_text))

    def test_analysis_sanitized_trait_rejects_internal_markers(self) -> None:
        self.assertFalse(
            evaluator.score_trait(
                "analysis_sanitized",
                "Ket qua",
                "<|channel|>analysis<|message|>leak<|return|>",
            )
        )


class TrainDryRunSmokeTest(unittest.TestCase):
    def test_dry_run_succeeds_without_transformers(self) -> None:
        stdout = io.StringIO()
        argv = [
            "train_transformers_deepthinkingflow_lora.py",
            "--config",
            str((ROOT_DIR / "training" / "DeepThinkingFlow-lora" / "config.example.json").resolve()),
            "--dry-run",
        ]

        with mock.patch.object(sys, "argv", argv):
            with contextlib.redirect_stdout(stdout):
                result = train_script.main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["tokenizer_precheck"], "ok")
        self.assertEqual(
            payload["behavior_bundle_dir"],
            "behavior/DeepThinkingFlow",
        )
        self.assertEqual(
            payload["skill_eval_cases_path"],
            "behavior/DeepThinkingFlow/evals/skill_compliance_following.jsonl",
        )
        self.assertIn("dependency_status", payload)
        self.assertTrue(payload["dependency_status"]["transformers"])

    def test_target_module_coverage_helpers_detect_missing_targets(self) -> None:
        class FakeParam:
            def __init__(self, numel: int, requires_grad: bool) -> None:
                self._numel = numel
                self.requires_grad = requires_grad

            def numel(self) -> int:
                return self._numel

        class FakeModel:
            def named_modules(self):
                return iter(
                    [
                        ("model.layers.0.self_attn.q_proj", object()),
                        ("model.layers.0.self_attn.k_proj", object()),
                        ("model.layers.0.self_attn.v_proj", object()),
                    ]
                )

            def parameters(self):
                return iter([FakeParam(10, True), FakeParam(90, False)])

        coverage = train_script.inspect_target_module_coverage(
            FakeModel(),
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.assertEqual(coverage["total_matches"], 3)
        self.assertEqual(coverage["missing_targets"], ["o_proj"])

        trainable = train_script.count_trainable_parameters(FakeModel())
        self.assertEqual(trainable["trainable_params"], 10)
        self.assertEqual(trainable["total_params"], 100)

    def test_validate_config_rejects_duplicate_target_modules(self) -> None:
        config = train_script.normalize_config(
            {
                "model_name_or_path": MODEL_DIR,
                "dataset_path": str((ROOT_DIR / "behavior" / "DeepThinkingFlow" / "training" / "harmony_sft_plus_skill_compliance_vi.train.jsonl").resolve()),
                "output_dir": str((ROOT_DIR / "out" / "dup-target-test").resolve()),
                "bf16": True,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "max_seq_length": 256,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "q_proj"],
                "reasoning_effort": "high",
            }
        )
        with self.assertRaises(ValueError):
            train_script.validate_config(config)

    def test_ensure_disjoint_splits_rejects_overlap(self) -> None:
        row = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
        with self.assertRaises(ValueError):
            train_script.ensure_disjoint_splits([row], [row])

    def test_staged_train_reports_missing_external_inputs_early(self) -> None:
        config = {
            "dataset_path": "data/external-train.jsonl",
            "eval_dataset_path": "data/external-eval.jsonl",
        }
        with self.assertRaises(SystemExit) as ctx:
            staged_train.ensure_stage_inputs_exist(config, Path("training/DeepThinkingFlow-lora/config.external-local-safe.stage1.json"))
        self.assertIn("Build the dataset assets first", str(ctx.exception))


class ProjectPreflightSmokeTest(unittest.TestCase):
    def test_project_preflight_returns_schema_and_ready_block(self) -> None:
        stdout = io.StringIO()
        argv = [
            "preflight_deepthinkingflow_project.py",
            "--bundle",
            BUNDLE_DIR,
            "--model-dir",
            MODEL_DIR,
            "--training-config",
            str((ROOT_DIR / "training" / "DeepThinkingFlow-lora" / "config.example.json").resolve()),
        ]

        with mock.patch.object(sys, "argv", argv):
            with contextlib.redirect_stdout(stdout):
                result = preflight_project.main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["schema_version"], "dtf-project-preflight/v1")
        self.assertIn("bundle_validation", payload)
        self.assertIn("training_feasibility", payload)
        self.assertIn("ready", payload)
        self.assertIn("status", payload)
        self.assertIn("external_runtime_status", payload["status"])
        self.assertFalse(payload["claim_boundary"]["raw_base_checkpoint_can_be_described_as_skill_aligned"])
        self.assertTrue(payload["claim_boundary"]["weight_level_adherence_requires_training_artifacts"])


class DoctorSmokeTest(unittest.TestCase):
    def test_doctor_returns_precondition_failed_when_host_is_not_ready(self) -> None:
        stdout = io.StringIO()
        argv = [
            "doctor_deepthinkingflow.py",
            "--bundle",
            BUNDLE_DIR,
            "--model-dir",
            MODEL_DIR,
            "--training-config",
            str((ROOT_DIR / "training" / "DeepThinkingFlow-lora" / "config.example.json").resolve()),
        ]

        with mock.patch.object(sys, "argv", argv):
            with contextlib.redirect_stdout(stdout):
                result = doctor_script.main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(result, 6)
        self.assertEqual(payload["schema_version"], "dtf-doctor-report/v1")
        self.assertIn("doctor", payload)
        self.assertTrue(payload["doctor"]["project_ready_for_runtime_only_release"])
        self.assertFalse(payload["doctor"]["project_ready_for_local_host_training"])
        self.assertTrue(payload["doctor"]["issues"])


class TinySmokeReleaseLaneTest(unittest.TestCase):
    def test_tiny_smoke_release_lane_orchestrates_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_config = Path(tmpdir) / "config.tiny.json"
            artifact_report = Path(tmpdir) / "artifacts.json"
            verify_report = Path(tmpdir) / "verify.json"
            release_manifest = Path(tmpdir) / "release.json"
            tiny_config.write_text(
                json.dumps(
                    {
                        "model_name_or_path": "runtime/transformers/DeepThinkingFlow-tiny-smoke",
                        "output_dir": str(Path(tmpdir) / "adapter"),
                        "dataset_path": "behavior/DeepThinkingFlow/training/harmony_sft_plus_skill_compliance_vi.train.jsonl",
                        "eval_dataset_path": "behavior/DeepThinkingFlow/training/harmony_sft_plus_skill_compliance_vi.eval.jsonl",
                        "behavior_bundle_dir": "behavior/DeepThinkingFlow",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            calls: list[list[str]] = []

            def fake_run(command: list[str], label: str) -> dict[str, object]:
                calls.append(command)
                if "report_deepthinkingflow_artifacts.py" in command[1]:
                    artifact_report.write_text("{}", encoding="utf-8")
                elif "verify_deepthinkingflow_project.py" in command[1]:
                    verify_report.write_text("{}", encoding="utf-8")
                elif "build_release_manifest.py" in command[1]:
                    release_manifest.write_text("{}", encoding="utf-8")
                return {"command": command, "stdout": "", "stderr": ""}

            stdout = io.StringIO()
            argv = [
                "run_tiny_smoke_release_lane.py",
                "--tiny-config",
                str(tiny_config),
                "--artifact-report",
                str(artifact_report),
                "--verify-report",
                str(verify_report),
                "--release-manifest",
                str(release_manifest),
            ]
            with mock.patch.object(tiny_release_lane, "run_command", side_effect=fake_run):
                with mock.patch.object(sys, "argv", argv):
                    with contextlib.redirect_stdout(stdout):
                        result = tiny_release_lane.main()

            payload = json.loads(stdout.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(payload["schema_version"], "dtf-tiny-smoke-release-lane/v1")
            self.assertEqual(len(calls), 4)
            self.assertEqual(payload["artifact_report"], str(artifact_report))
            self.assertEqual(payload["verify_report"], str(verify_report))
            self.assertEqual(payload["release_manifest"], str(release_manifest))


class ProjectVerifySmokeTest(unittest.TestCase):
    def test_verify_project_supports_skip_tests(self) -> None:
        stdout = io.StringIO()
        argv = [
            "verify_deepthinkingflow_project.py",
            "--bundle",
            BUNDLE_DIR,
            "--model-dir",
            MODEL_DIR,
            "--training-config",
            str((ROOT_DIR / "training" / "DeepThinkingFlow-lora" / "config.example.json").resolve()),
            "--skip-tests",
        ]

        with mock.patch.object(sys, "argv", argv):
            with contextlib.redirect_stdout(stdout):
                result = verify_project.main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(payload["schema_version"], "dtf-verify-report/v2")
        self.assertIn("generated_at_utc", payload)
        self.assertIn("environment", payload)
        self.assertIn("commands", payload)
        self.assertIn("claim_gate", payload)
        self.assertTrue(payload["verified"]["bundle_valid"])
        self.assertTrue(payload["verified"]["preflight_ran"])
        self.assertTrue(payload["verified"]["tests_passed"])
        self.assertTrue(payload["verified"]["claim_gate_passed"])
        self.assertFalse(payload["project_preflight"]["claim_boundary"]["raw_base_checkpoint_can_be_described_as_skill_aligned"])

    def test_verify_project_fails_claim_gate_when_training_ready_missing_artifacts(self) -> None:
        stdout = io.StringIO()
        argv = [
            "verify_deepthinkingflow_project.py",
            "--bundle",
            BUNDLE_DIR,
            "--model-dir",
            MODEL_DIR,
            "--training-config",
            str((ROOT_DIR / "training" / "DeepThinkingFlow-lora" / "config.example.json").resolve()),
            "--skip-tests",
            "--require-claim-level",
            "training-ready",
        ]

        with mock.patch.object(sys, "argv", argv):
            with contextlib.redirect_stdout(stdout):
                result = verify_project.main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(result, 4)
        self.assertFalse(payload["claim_gate"]["passed"])
        self.assertIn("below required", payload["claim_gate"]["reasons"][0])


class ReleaseManifestSmokeTest(unittest.TestCase):
    def test_build_release_manifest_from_verify_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            verify_path = Path(tmpdir) / "verify.json"
            output_path = Path(tmpdir) / "release.json"
            verify_payload = {
                "schema_version": "dtf-verify-report/v2",
                "verified": {
                    "bundle_valid": True,
                    "preflight_ran": True,
                    "tests_passed": True,
                    "claim_gate_passed": True,
                },
                "project_preflight": {
                    "ready": {
                        "bundle_valid": True,
                        "inference_soft_gate_clear": False,
                        "training_soft_gate_clear": False,
                        "training_locally_feasible": False,
                    }
                },
                "claim_gate": {"passed": True},
            }
            verify_path.write_text(json.dumps(verify_payload, ensure_ascii=False), encoding="utf-8")

            stdout = io.StringIO()
            argv = [
                "build_release_manifest.py",
                "--verify-report",
                str(verify_path),
                "--output",
                str(output_path),
                "--release-id",
                "test-release",
            ]
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(stdout):
                    result = release_manifest.main()

            payload = json.loads(stdout.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(payload["schema_version"], "dtf-release-manifest/v1")
            self.assertEqual(payload["release_id"], "test-release")
            self.assertEqual(payload["release_state"]["claim_level"], "runtime-only")
            self.assertTrue(payload["release_state"]["claim_gate_passed"])
            self.assertTrue(payload["release_state"]["release_candidate"])
            self.assertFalse(payload["release_state"]["local_host_ready"])
            self.assertEqual(payload["verify_report_ref"]["path"], str(verify_path))
            self.assertTrue(payload["verify_report_ref"]["sha256"])
            self.assertTrue(output_path.is_file())

    def test_build_release_manifest_rejects_failed_claim_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            verify_path = Path(tmpdir) / "verify.json"
            verify_path.write_text(
                json.dumps(
                    {
                        "schema_version": "dtf-verify-report/v2",
                        "verified": {
                            "bundle_valid": True,
                            "preflight_ran": True,
                            "tests_passed": True,
                            "claim_gate_passed": False,
                        },
                        "claim_gate": {
                            "passed": False,
                            "reasons": ["Artifact lineage is incomplete."],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            argv = [
                "build_release_manifest.py",
                "--verify-report",
                str(verify_path),
                "--output",
                str(Path(tmpdir) / "release.json"),
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit) as ctx:
                    release_manifest.main()

        self.assertIn("claim gate", str(ctx.exception))


class StagedTrainingConfigResolutionTest(unittest.TestCase):
    def test_build_effective_config_resolves_latest_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            checkpoint_dir = output_dir / "checkpoint-12"
            checkpoint_dir.mkdir(parents=True)
            config = {
                "output_dir": str(output_dir),
                "resume_from_checkpoint": "latest",
            }

            effective = staged_train.build_effective_config(config)

            self.assertEqual(config["resume_from_checkpoint"], "latest")
            self.assertEqual(effective["resume_from_checkpoint"], str(checkpoint_dir.resolve()))

    def test_write_effective_config_creates_temp_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = Path(tmpdir) / "config.stage.json"
            stage_path.write_text('{"resume_from_checkpoint":"latest"}\n', encoding="utf-8")

            temp_path = staged_train.write_effective_config(stage_path, {"resume_from_checkpoint": "checkpoint-1"})

            self.assertNotEqual(temp_path, stage_path)
            self.assertEqual(json.loads(stage_path.read_text(encoding="utf-8"))["resume_from_checkpoint"], "latest")
            self.assertEqual(json.loads(temp_path.read_text(encoding="utf-8"))["resume_from_checkpoint"], "checkpoint-1")


class PreparedChatExportSmokeTest(unittest.TestCase):
    def test_export_prepared_chat_jsonl_writes_rows(self) -> None:
        try:
            from datasets import Dataset
        except Exception:
            self.skipTest("datasets is not installed in the current environment")

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "prepared"
            output_jsonl = Path(tmpdir) / "prepared.jsonl"
            Dataset.from_list(
                [
                    {
                        "messages": [
                            {"role": "user", "content": "hello"},
                            {"role": "assistant", "content": "world"},
                        ],
                        "text": "hello world",
                        "source_dataset": "unit-test",
                    }
                ]
            ).save_to_disk(str(dataset_dir))

            stdout = io.StringIO()
            argv = [
                "export_prepared_chat_jsonl.py",
                "--input-dir",
                str(dataset_dir),
                "--output-jsonl",
                str(output_jsonl),
            ]
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(stdout):
                    result = export_chat_jsonl.main()

            payload = json.loads(stdout.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(payload["rows_exported"], 1)
            self.assertTrue(output_jsonl.is_file())
            rows = output_jsonl.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            exported = json.loads(rows[0])
            self.assertEqual(exported["source_dataset"], "unit-test")
            self.assertEqual(exported["messages"][-1]["content"], "world")


class ExternalTrainingBundleSmokeTest(unittest.TestCase):
    def test_build_external_training_bundle_writes_train_eval_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_a = Path(tmpdir) / "a.jsonl"
            input_b = Path(tmpdir) / "b.jsonl"
            train_out = Path(tmpdir) / "external-train.jsonl"
            eval_out = Path(tmpdir) / "external-eval.jsonl"

            rows_a = [
                {"messages": [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}], "source_dataset": "a"},
                {"messages": [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}], "source_dataset": "a"},
            ]
            rows_b = [
                {"messages": [{"role": "user", "content": "u3"}, {"role": "assistant", "content": "a3"}], "source_dataset": "b"},
                {"messages": [{"role": "user", "content": "u4"}, {"role": "assistant", "content": "a4"}], "source_dataset": "b"},
            ]
            input_a.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows_a) + "\n", encoding="utf-8")
            input_b.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows_b) + "\n", encoding="utf-8")

            stdout = io.StringIO()
            argv = [
                "build_external_training_bundle.py",
                "--input-jsonl",
                str(input_a),
                "--input-jsonl",
                str(input_b),
                "--train-output",
                str(train_out),
                "--eval-output",
                str(eval_out),
                "--eval-ratio",
                "0.25",
            ]
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(stdout):
                    result = build_external_bundle.main()

            payload = json.loads(stdout.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(payload["train_rows"], 3)
            self.assertEqual(payload["eval_rows"], 1)
            self.assertTrue(train_out.is_file())
            self.assertTrue(eval_out.is_file())


class TrainingAssetBuilderTest(unittest.TestCase):
    def test_builder_creates_disjoint_fixed_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = Path(tmpdir) / "behavior" / "DeepThinkingFlow"
            training = bundle / "training"
            training.mkdir(parents=True, exist_ok=True)

            base_all = []
            for idx in range(4):
                base_all.append(
                    {
                        "messages": [
                            {"role": "system", "content": "sys"},
                            {"role": "user", "content": f"base-{idx}"},
                            {"role": "assistant", "content": f"ans-{idx}"},
                        ]
                    }
                )
            base_train = base_all[:3]
            base_eval = base_all[3:]

            skill_all = []
            for category in asset_builder.REQUIRED_SKILL_CATEGORIES:
                for idx in range(3):
                    skill_all.append(
                        {
                            "category": category,
                            "messages": [
                                {"role": "system", "content": "sys"},
                                {"role": "user", "content": f"{category}-{idx}"},
                                {"role": "assistant", "content": f"{category}-ans-{idx}"},
                            ],
                        }
                    )

            for name, rows in {
                "harmony_sft_vi.jsonl": base_all,
                "harmony_sft_vi.train.jsonl": base_train,
                "harmony_sft_vi.eval.jsonl": base_eval,
                "harmony_sft_skill_compliance_vi.jsonl": skill_all,
            }.items():
                (training / name).write_text(
                    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                    encoding="utf-8",
                )

            argv = [
                "prepare_deepthinkingflow_training_assets.py",
                "--bundle",
                str(bundle),
                "--skill-eval-per-category",
                "1",
            ]
            stdout = io.StringIO()
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(stdout):
                    result = asset_builder.main()

            summary = json.loads(stdout.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(summary["skill_compliance"]["train"], 8)
            self.assertEqual(summary["skill_compliance"]["eval"], 4)
            self.assertEqual(summary["combined"]["train"], 11)
            self.assertEqual(summary["combined"]["eval"], 5)

            skill_train_rows = bundle_validator.read_jsonl(training / "harmony_sft_skill_compliance_vi.train.jsonl")
            skill_eval_rows = bundle_validator.read_jsonl(training / "harmony_sft_skill_compliance_vi.eval.jsonl")
            combined_train_rows = bundle_validator.read_jsonl(training / "harmony_sft_plus_skill_compliance_vi.train.jsonl")
            combined_eval_rows = bundle_validator.read_jsonl(training / "harmony_sft_plus_skill_compliance_vi.eval.jsonl")

            skill_train_hashes = {bundle_validator.canonical_messages_hash(row["messages"]) for row in skill_train_rows}
            skill_eval_hashes = {bundle_validator.canonical_messages_hash(row["messages"]) for row in skill_eval_rows}
            self.assertTrue(skill_train_hashes.isdisjoint(skill_eval_hashes))
            self.assertEqual(len(combined_train_rows), 11)
            self.assertEqual(len(combined_eval_rows), 5)


class SafetensorsInspectorTest(unittest.TestCase):
    def test_inspector_reports_raw_checkpoint_and_config_match(self) -> None:
        def fake_tensor(dtype: str, shape: list[int]) -> dict[str, object]:
            return {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}

        header = {
            "embedding.weight": fake_tensor("BF16", [16, 8]),
            "block.0.attn.norm.scale": fake_tensor("BF16", [8]),
            "block.0.attn.out.bias": fake_tensor("BF16", [8]),
            "block.0.attn.out.weight": fake_tensor("BF16", [8, 8]),
            "block.0.attn.qkv.bias": fake_tensor("BF16", [16]),
            "block.0.attn.qkv.weight": fake_tensor("BF16", [16, 8]),
            "block.0.attn.sinks": fake_tensor("BF16", [2]),
            "block.0.mlp.gate.bias": fake_tensor("BF16", [3]),
            "block.0.mlp.gate.weight": fake_tensor("BF16", [3, 8]),
            "block.0.mlp.mlp1_bias": fake_tensor("BF16", [3, 8]),
            "block.0.mlp.mlp1_weight.blocks": fake_tensor("U8", [3, 8, 1, 16]),
            "block.0.mlp.mlp1_weight.scales": fake_tensor("U8", [3, 8, 1]),
            "block.0.mlp.mlp2_bias": fake_tensor("BF16", [3, 4]),
            "block.0.mlp.mlp2_weight.blocks": fake_tensor("U8", [3, 4, 1, 16]),
            "block.0.mlp.mlp2_weight.scales": fake_tensor("U8", [3, 4, 1]),
            "block.0.mlp.norm.scale": fake_tensor("BF16", [8]),
            "norm.scale": fake_tensor("BF16", [8]),
            "unembedding.weight": fake_tensor("BF16", [16, 8]),
        }
        config = {
            "architectures": ["GptOssForCausalLM"],
            "model_type": "gpt_oss",
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "num_local_experts": 3,
            "num_experts_per_tok": 1,
            "experts_per_token": 1,
            "layer_types": ["full_attention"],
            "rope_theta": 10000,
            "sliding_window": 16,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_path = root / "model.safetensors"
            config_path = root / "config.json"
            dtypes_path = root / "dtypes.json"

            header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            model_path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + (b"\x00" * 64))
            config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            dtypes_path.write_text(
                json.dumps(
                    {
                        "block.0.mlp.mlp1_weight.blocks": "FP4",
                        "block.0.mlp.mlp1_weight.scales": "UE8",
                        "block.0.mlp.mlp2_weight.blocks": "FP4",
                        "block.0.mlp.mlp2_weight.scales": "UE8",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary, report = inspector.inspect_model(model_path, config_path, tensor_limit=5)

        self.assertEqual(summary["architecture_validation"]["status"], "pass")
        self.assertTrue(summary["single_raw_safetensors"])
        self.assertIn("Trong file này không có gì", report)
        self.assertEqual(summary["logical_dtype_counts"]["FP4"], 2)
        self.assertEqual(summary["logical_dtype_counts"]["UE8"], 2)


class ArtifactReportSmokeTest(unittest.TestCase):
    def test_artifact_report_classifies_claim_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_weights = root / "model.safetensors"
            eval_output = root / "eval.json"
            compare_report = root / "compare.json"
            adapter_dir = root / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            base_weights.write_bytes(b"weights")
            eval_output.write_text("{}", encoding="utf-8")
            compare_report.write_text("{}", encoding="utf-8")

            summary = {
                "base_weights": artifact_report.collect_path_report(base_weights),
                "adapter_dir": artifact_report.collect_path_report(adapter_dir),
                "eval_output": artifact_report.collect_path_report(eval_output),
                "compare_report": artifact_report.collect_path_report(compare_report),
            }
            claim_level = artifact_report.detect_claim_level(
                summary["base_weights"],
                summary["adapter_dir"],
                summary["eval_output"],
                summary["compare_report"],
            )

        self.assertEqual(claim_level, "learned-only-after-training")

    def test_artifact_report_builds_lineage_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_dir = root / "behavior" / "DeepThinkingFlow"
            training_dir = bundle_dir / "training"
            bundle_dir.mkdir(parents=True)
            training_dir.mkdir(parents=True)
            (bundle_dir / "profile.json").write_text("{}", encoding="utf-8")
            (bundle_dir / "system_prompt.txt").write_text("system", encoding="utf-8")

            train_dataset = training_dir / "train.jsonl"
            eval_dataset = training_dir / "eval.jsonl"
            train_dataset.write_text("{}\n", encoding="utf-8")
            eval_dataset.write_text("{}\n", encoding="utf-8")

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset_path": str(train_dataset),
                        "eval_dataset_path": str(eval_dataset),
                        "behavior_bundle_dir": str(bundle_dir),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            lineage = artifact_report.build_lineage_status(
                training_config_payload=artifact_report.load_training_config(config_path),
                train_dataset=artifact_report.collect_path_report(train_dataset),
                eval_dataset=artifact_report.collect_path_report(eval_dataset),
                behavior_bundle=artifact_report.collect_path_report(bundle_dir),
                base_weights={"path": str(root / "model.safetensors")},
                adapter_dir={"path": str(root / "adapter")},
                eval_output={"path": str(root / "eval.json")},
            )

        self.assertTrue(lineage["config_dataset_match"])
        self.assertTrue(lineage["config_eval_dataset_match"])
        self.assertTrue(lineage["config_bundle_match"])
        self.assertTrue(lineage["lineage_complete_for_training_claim"])
        self.assertTrue(lineage["lineage_complete_for_learned_claim"])


class EnvHelpersTest(unittest.TestCase):
    def test_dependency_status_detects_transformers(self) -> None:
        status = dtf_env.detect_dependency_status()
        self.assertIn("transformers", status)


class TrainingPreflightTest(unittest.TestCase):
    def test_classifies_20b_config_as_not_local_safe(self) -> None:
        config = {
            "model_name_or_path": "runtime/transformers/DeepThinkingFlow",
            "use_qlora": False,
        }
        env = {
            "memory": {
                "mem_total_bytes": 8 * 1024**3,
                "mem_available_bytes": 2 * 1024**3,
                "swap_total_bytes": 8 * 1024**3,
                "swap_free_bytes": 2 * 1024**3,
            },
            "gpu": {
                "has_nvidia_smi": False,
                "nvidia_smi_path": "",
            },
        }
        model_info = {
            "resolved_model_dir": "/tmp/DeepThinkingFlow",
            "weight_file_bytes": 13 * 1024**3,
        }

        result = preflight_train.classify_feasibility(config, env, model_info)

        self.assertFalse(result["can_attempt_local_training"])
        self.assertTrue(any("20B" in reason or "DeepThinkingFlow 20B" in reason for reason in result["reasons"]))


class CompileBundleTest(unittest.TestCase):
    def test_compiler_creates_compact_prompt_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = Path(tmpdir) / "behavior" / "DeepThinkingFlow"
            bundle.mkdir(parents=True, exist_ok=True)
            (bundle / "profile.json").write_text(
                json.dumps({"name": "DeepThinkingFlow", "target_model": "DeepThinkingFlow", "compliance_model": {"weight_level_adherence_requires_training": True}}, ensure_ascii=False),
                encoding="utf-8",
            )
            (bundle / "system_prompt.txt").write_text(
                "<identity>You are DeepThinkingFlow.</identity>\n<hard_rules>\n- rule a\n- rule b\n</hard_rules>\n<task_classifier>\n- explain\n- debug\n</task_classifier>\n<depth_policy>\n- Quick: short\n</depth_policy>\n<output_policy>\n- Answer first\n</output_policy>\n<local_model_guidance>\n- Keep compact\n</local_model_guidance>\n<quality_bar>\n- Be concrete\n</quality_bar>\n",
                encoding="utf-8",
            )

            pack = compile_bundle.build_pack(bundle)
            self.assertIn("compact_system_prompt", pack)
            self.assertIn("RULES=", pack["compact_system_prompt"])
            self.assertLess(len(pack["compact_system_prompt"]), 1000)
            self.assertIn("runtime_pack", pack)
            self.assertIn("runtime_pack_text", pack)
            self.assertIn("STACK=", pack["runtime_pack_text"])


if __name__ == "__main__":
    unittest.main()
