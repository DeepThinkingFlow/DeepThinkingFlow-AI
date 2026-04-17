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
import preflight_deepthinkingflow_training as preflight_train
import deepthinkingflow_runtime as runtime
import evaluate_reasoning_outputs as evaluator
import inspect_safetensors_model as inspector
import prepare_deepthinkingflow_training_assets as asset_builder
import report_deepthinkingflow_artifacts as artifact_report
import render_transformers_deepthinkingflow_prompt as render_prompt
import run_transformers_deepthinkingflow as run_script
import train_transformers_deepthinkingflow_lora as train_script
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
