"""Unit tests for training utilities."""

import os
import tempfile
import unittest
from unittest import mock

from test.training._deps import missing_optional_packages


class TestModelDirectoryUtils(unittest.TestCase):
    """Test model directory utility functions."""

    def test_check_model_exists_with_existing_model(self):
        """Test check_model_exists returns True for existing model."""
        from src.training import check_model_exists

        # Create temporary directory with a model file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake model file
            model_file = os.path.join(tmpdir, "pytorch_model.bin")
            with open(model_file, "w") as f:
                f.write("fake model")

            # Should return True
            self.assertTrue(check_model_exists(tmpdir))

    def test_check_model_exists_with_safetensors(self):
        """Test check_model_exists returns True for safetensors model."""
        from src.training import check_model_exists

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a safetensors file
            model_file = os.path.join(tmpdir, "model.safetensors")
            with open(model_file, "w") as f:
                f.write("fake model")

            self.assertTrue(check_model_exists(tmpdir))

    def test_check_model_exists_with_nonexistent_directory(self):
        """Test check_model_exists returns False for non-existent directory."""
        from src.training import check_model_exists

        self.assertFalse(check_model_exists("/nonexistent/path"))

    def test_check_model_exists_with_empty_directory(self):
        """Test check_model_exists returns False for empty directory."""
        from src.training import check_model_exists

        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(check_model_exists(tmpdir))


class TestUnlearningMethodParsing(unittest.TestCase):
    """Test unlearning method parsing functions."""

    def test_parse_valid_methods(self):
        """Test parsing of all valid unlearning methods."""
        from src.training.methods import UNLEARNING_METHODS, parse_unlearning_method

        for method in UNLEARNING_METHODS:
            unlearning_type, reg_type = parse_unlearning_method(method)

            # Check format
            self.assertIn(unlearning_type, ["ga", "npo", "idk", "dpo"])
            self.assertIn(reg_type, ["gd", "kl"])

    def test_parse_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        from src.training.methods import parse_unlearning_method

        with self.assertRaises(ValueError) as context:
            parse_unlearning_method("invalid_method")

        self.assertIn("Unknown unlearning method", str(context.exception))

    def test_parse_malformed_method_raises_error(self):
        """Test that malformed method string raises ValueError."""
        from src.training.methods import parse_unlearning_method

        # Method with wrong format
        with self.assertRaises(ValueError):
            parse_unlearning_method("ga")  # Missing regularization part

        with self.assertRaises(ValueError):
            parse_unlearning_method("ga_gd_extra")  # Too many parts


class TestMethodRequirements(unittest.TestCase):
    """Test method requirement checking functions."""

    def test_needs_idk_dataset_for_all_methods(self):
        """Test IDK dataset requirement for all methods."""
        from src.training.methods import UNLEARNING_METHODS, needs_idk_dataset

        expected_idk_methods = {"idk_gd", "idk_kl", "dpo_gd", "dpo_kl"}
        expected_no_idk_methods = {"ga_gd", "ga_kl", "npo_gd", "npo_kl"}

        for method in UNLEARNING_METHODS:
            if method in expected_idk_methods:
                self.assertTrue(
                    needs_idk_dataset(method),
                    f"{method} should need IDK dataset",
                )
            elif method in expected_no_idk_methods:
                self.assertFalse(
                    needs_idk_dataset(method),
                    f"{method} should not need IDK dataset",
                )

    def test_needs_reference_model_for_all_methods(self):
        """Test reference model requirement for all methods."""
        from src.training.methods import UNLEARNING_METHODS, needs_reference_model

        # All KL methods + all DPO methods need reference model
        expected_ref_methods = {"ga_kl", "npo_gd", "npo_kl", "idk_kl", "dpo_kl", "dpo_gd"}
        expected_no_ref_methods = {"ga_gd", "idk_gd"}

        for method in UNLEARNING_METHODS:
            if method in expected_ref_methods:
                self.assertTrue(
                    needs_reference_model(method),
                    f"{method} should need reference model",
                )
            elif method in expected_no_ref_methods:
                self.assertFalse(
                    needs_reference_model(method),
                    f"{method} should not need reference model",
                )


class TestBaseModelDirComponent(unittest.TestCase):
    """Test base model directory name generation."""

    def test_known_model_mappings(self):
        """Test that known models map to expected directory names."""
        from src.training import get_base_model_dir_component

        known_mappings = {
            "meta-llama/Llama-3.2-1B": "llama-3.2-1b",
            "meta-llama/Llama-3.2-1B-Instruct": "llama-3.2-1b",
            "meta-llama/Llama-3.2-3B": "llama-3.2-3b",
            "meta-llama/Llama-3.1-8B": "llama-3.1-8b",
            "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",
        }

        for model_name, expected_dir in known_mappings.items():
            result = get_base_model_dir_component(model_name)
            self.assertEqual(
                result,
                expected_dir,
                f"Model {model_name} should map to {expected_dir}",
            )

    def test_unknown_model_sanitization(self):
        """Test that unknown models are properly sanitized."""
        from src.training import get_base_model_dir_component

        # Test with special characters
        result = get_base_model_dir_component("some/model/with/slashes")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

        # Should only contain alphanumeric and single dashes
        self.assertNotIn("/", result)
        self.assertNotIn("--", result)

    def test_sanitization_removes_consecutive_dashes(self):
        """Test that consecutive dashes are removed."""
        from src.training import get_base_model_dir_component

        result = get_base_model_dir_component("some//model//name")
        self.assertNotIn("--", result)

    def test_sanitization_strips_dashes(self):
        """Test that leading/trailing dashes are removed."""
        from src.training import get_base_model_dir_component

        result = get_base_model_dir_component("/model/")
        self.assertFalse(result.startswith("-"))
        self.assertFalse(result.endswith("-"))


class TestMethodDescriptions(unittest.TestCase):
    """Test that method descriptions are complete and valid."""

    def test_all_methods_have_descriptions(self):
        """Test that all methods have descriptions."""
        from src.training.methods import METHOD_DESCRIPTIONS, UNLEARNING_METHODS

        for method in UNLEARNING_METHODS:
            self.assertIn(
                method,
                METHOD_DESCRIPTIONS,
                f"Method {method} should have a description",
            )

    def test_descriptions_are_non_empty(self):
        """Test that all descriptions are non-empty strings."""
        from src.training.methods import METHOD_DESCRIPTIONS

        for method, description in METHOD_DESCRIPTIONS.items():
            self.assertIsInstance(description, str)
            self.assertGreater(
                len(description),
                0,
                f"Description for {method} should not be empty",
            )

    def test_no_extra_descriptions(self):
        """Test that there are no extra descriptions."""
        from src.training.methods import METHOD_DESCRIPTIONS, UNLEARNING_METHODS

        # All descriptions should be for valid methods
        for method in METHOD_DESCRIPTIONS.keys():
            self.assertIn(
                method,
                UNLEARNING_METHODS,
                f"Description exists for unknown method {method}",
            )


class TestTrainingDeviceMapResolution(unittest.TestCase):
    """Test device-map normalization for Lightning-managed training."""

    def test_single_gpu_auto_device_map_is_pinned(self):
        """Auto-sharding should be disabled for single-GPU Lightning runs."""
        from src.training.device_map import resolve_training_device_map

        resolved = resolve_training_device_map(
            requested_device_map="auto",
            accelerator="gpu",
            devices=1,
            cuda_available=True,
            cuda_device_count=2,
        )

        self.assertEqual(resolved, "cuda:0")

    def test_cpu_trainer_forces_cpu_device_map(self):
        """CPU trainer runs should not auto-place models onto visible GPUs."""
        from src.training.device_map import resolve_training_device_map

        resolved = resolve_training_device_map(
            requested_device_map="auto",
            accelerator="cpu",
            devices=1,
            cuda_available=True,
            cuda_device_count=2,
        )

        self.assertEqual(resolved, "cpu")


class TestBuilderDeviceMapResolution(unittest.TestCase):
    """Integration-style checks for builder-level device-map handling."""

    @classmethod
    def setUpClass(cls) -> None:
        missing = missing_optional_packages("lightning", "omegaconf", "torch")
        if missing:
            raise unittest.SkipTest(
                "Missing optional dependency(s): " + ", ".join(sorted(missing))
            )

    def test_builder_pins_single_gpu_auto_device_map(self):
        """The shared builder path should apply single-GPU normalization."""
        from omegaconf import OmegaConf

        from src.training.builders import resolve_model_device_map

        cfg = OmegaConf.create(
            {
                "model": {"device_map": "auto"},
                "trainer": {"accelerator": "gpu", "devices": 1},
            }
        )

        with mock.patch(
            "src.training.builders.torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "src.training.builders.torch.cuda.device_count",
            return_value=2,
        ):
            self.assertEqual(resolve_model_device_map(cfg), "cuda:0")


class TestUnlearningModelPathResolution(unittest.TestCase):
    """Test resolution precedence for unlearning model paths."""

    @classmethod
    def setUpClass(cls) -> None:
        missing = missing_optional_packages(
            "lightning", "omegaconf", "torch", "transformers"
        )
        if missing:
            raise unittest.SkipTest(
                "Missing optional dependency(s): " + ", ".join(sorted(missing))
            )

    def test_explicit_model_path_wins_over_trained_tag(self):
        """CLI-supplied model.path should not be overwritten by trained_tag."""
        from omegaconf import OmegaConf

        from src.training.builders import maybe_resolve_unlearning_model_path

        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "path": "/tmp/custom-model",
                    "trained_tag": "finetune_retain-tinyllama-qlora",
                }
            }
        )

        resolved = maybe_resolve_unlearning_model_path(cfg)

        self.assertEqual(resolved, "/tmp/custom-model")
        self.assertEqual(cfg.model.path, "/tmp/custom-model")

    def test_trained_tag_builds_default_model_path_when_path_missing(self):
        """trained_tag should synthesize the default model path when needed."""
        from omegaconf import OmegaConf

        from src.training.builders import maybe_resolve_unlearning_model_path

        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "path": None,
                    "trained_tag": "finetune_retain-tinyllama-qlora",
                }
            }
        )

        resolved = maybe_resolve_unlearning_model_path(cfg)

        expected = (
            "model/finetune_retain-tinyllama-qlora/finetune/"
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self.assertEqual(resolved, expected)
        self.assertEqual(cfg.model.path, expected)

    def test_missing_path_and_trained_tag_raises(self):
        """Unlearning still requires one source of model location."""
        from omegaconf import OmegaConf

        from src.training.builders import maybe_resolve_unlearning_model_path

        cfg = OmegaConf.create(
            {
                "model": {
                    "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "path": None,
                    "trained_tag": None,
                }
            }
        )

        with self.assertRaises(ValueError):
            maybe_resolve_unlearning_model_path(cfg)


class TestExternalLoggerProviderResolution(unittest.TestCase):
    """Test generic external tracking provider resolution."""

    @classmethod
    def setUpClass(cls) -> None:
        missing = missing_optional_packages("omegaconf")
        if missing:
            raise unittest.SkipTest(
                "Missing optional dependency(s): " + ", ".join(sorted(missing))
            )

    def test_tracking_provider_defaults_to_none(self):
        """No tracking config should disable external logging."""
        from omegaconf import OmegaConf

        from src.training.external_logging import resolve_external_logger_provider

        cfg = OmegaConf.create({})
        self.assertEqual(resolve_external_logger_provider(cfg), "none")

    def test_tracking_provider_prefers_explicit_tracking_config(self):
        """The generic tracking block should drive provider resolution."""
        from omegaconf import OmegaConf

        from src.training.external_logging import resolve_external_logger_provider

        cfg = OmegaConf.create(
            {
                "tracking": {"enabled": True, "provider": "mlflow"},
                "mlflow": {"enabled": True},
            }
        )
        self.assertEqual(resolve_external_logger_provider(cfg), "mlflow")

    def test_mlflow_config_is_recognized_without_tracking_block(self):
        """The MLflow config block alone should enable MLflow tracking."""
        from omegaconf import OmegaConf

        from src.training.external_logging import resolve_external_logger_provider

        cfg = OmegaConf.create({"mlflow": {"enabled": True}})
        self.assertEqual(resolve_external_logger_provider(cfg), "mlflow")

    def test_disabled_tracking_returns_none_provider(self):
        """Disabled tracking should prevent external logger creation."""
        from omegaconf import OmegaConf

        from src.training.external_logging import resolve_external_logger_provider

        cfg = OmegaConf.create({"tracking": {"enabled": False, "provider": "mlflow"}})
        self.assertEqual(resolve_external_logger_provider(cfg), "none")


class TestMlflowLoggerCreation(unittest.TestCase):
    """Test MLflow logger initialization on a local file backend."""

    @classmethod
    def setUpClass(cls) -> None:
        missing = missing_optional_packages("lightning", "mlflow", "omegaconf")
        if missing:
            raise unittest.SkipTest(
                "Missing optional dependency(s): " + ", ".join(sorted(missing))
            )

    def test_create_external_logger_with_mlflow_backend(self):
        """The generic external logger factory should create an MLflow logger."""
        from pathlib import Path

        from mlflow.tracking import MlflowClient
        from omegaconf import OmegaConf

        from src.training.external_logging import (
            create_external_logger,
            finalize_external_logger,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create(
                {
                    "project_path": tmpdir,
                    "task": "finetune",
                    "tag": "test",
                    "model": {"name": "sshleifer/tiny-gpt2"},
                    "data": {
                        "name": "local",
                        "train_split": "train",
                        "train_subset_name": "mini",
                    },
                    "training": {
                        "epochs": 1,
                        "train_batch_size": 2,
                        "gradient_accumulation_steps": 1,
                        "learning_rate": 1e-4,
                        "weight_decay": 0.01,
                        "warmup_ratio": 0.1,
                        "max_grad_norm": 1.0,
                    },
                    "tracking": {"enabled": True, "provider": "mlflow"},
                    "mlflow": {
                        "enabled": True,
                        "tracking_uri": None,
                        "artifact_root": os.path.join(tmpdir, "mlartifacts"),
                        "experiment_name": "RAQUEL-tests",
                        "run_name": "unit-test",
                        "tags": {"suite": "unit"},
                        "log_model_artifacts": False,
                        "log_generation_artifacts": False,
                        "log_run_artifacts": False,
                    },
                    "results": {"run_name": None},
                }
            )

            logger_instance = create_external_logger(cfg)
            self.assertIsNotNone(logger_instance)
            self.assertEqual(type(logger_instance).__name__, "MLFlowLogger")
            self.assertTrue(getattr(logger_instance, "run_id", None))
            client = MlflowClient(tracking_uri=f"sqlite:///{Path(tmpdir) / 'mlflow.db'}")
            experiment = client.get_experiment_by_name("RAQUEL-tests")
            self.assertIsNotNone(experiment)
            self.assertEqual(
                experiment.artifact_location,
                str((Path(tmpdir) / "mlartifacts").resolve()),
            )
            finalize_external_logger(logger_instance)

    def test_explicit_device_map_is_preserved(self):
        """Explicit user device maps should win over automatic normalization."""
        from src.training.device_map import resolve_training_device_map

        resolved = resolve_training_device_map(
            requested_device_map="cuda:1",
            accelerator="gpu",
            devices=1,
            cuda_available=True,
            cuda_device_count=2,
        )

        self.assertEqual(resolved, "cuda:1")

    def test_multi_gpu_auto_device_map_is_left_unchanged(self):
        """Multi-device runs keep explicit auto-placement semantics."""
        from src.training.device_map import resolve_training_device_map

        resolved = resolve_training_device_map(
            requested_device_map="auto",
            accelerator="gpu",
            devices=2,
            cuda_available=True,
            cuda_device_count=2,
        )

        self.assertEqual(resolved, "auto")


class TestTrainingRuntimeSetup(unittest.TestCase):
    """Test runtime environment setup helpers."""

    @classmethod
    def setUpClass(cls) -> None:
        missing = missing_optional_packages("lightning", "omegaconf", "torch")
        if missing:
            raise unittest.SkipTest(
                "Missing optional dependency(s): " + ", ".join(sorted(missing))
            )

    def test_setup_environment_honors_zero_seed(self):
        """A seed value of zero should still enable deterministic setup."""
        from omegaconf import OmegaConf

        from src.training.runtime import setup_environment

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create(
                {
                    "seed": 0,
                    "output": {"dir": os.path.join(tmpdir, "model")},
                    "checkpoint": {"dirpath": os.path.join(tmpdir, "ckpts")},
                }
            )

            with mock.patch(
                "src.training.runtime.pl.seed_everything"
            ) as seed_everything_mock, mock.patch(
                "src.training.runtime.torch.cuda.is_available", return_value=False
            ):
                setup_environment(cfg)

            seed_everything_mock.assert_called_once_with(0)
            self.assertTrue(os.path.isdir(cfg.output.dir))
            self.assertTrue(os.path.isdir(cfg.checkpoint.dirpath))


class TestEvaluationLoaderWrapper(unittest.TestCase):
    """Test that evaluation scripts delegate to the shared model loader."""

    def test_load_fine_tuned_model_uses_shared_loader_for_cpu_flow(self):
        """device_map_auto=False should preserve CPU loading semantics."""
        from script.evaluation.utils import load_fine_tuned_model

        sentinel_model = object()
        sentinel_tokenizer = object()
        with mock.patch(
            "script.evaluation.utils.load_model_and_tokenizer",
            return_value=(sentinel_model, sentinel_tokenizer),
        ) as loader_mock:
            model, tokenizer = load_fine_tuned_model(
                "adapter-dir",
                base_model_name="base-model",
                device_map_auto=False,
                quantize_4bit=False,
                as_trainable=True,
            )

        loader_mock.assert_called_once_with(
            "adapter-dir",
            device_map=None,
            use_fp16=False,
            ddp_mode=False,
            quantize_4bit=False,
            lora=None,
            base_model_name_for_adapters="base-model",
            adapter_trainable=True,
        )
        self.assertIs(model, sentinel_model)
        self.assertIs(tokenizer, sentinel_tokenizer)

    def test_load_fine_tuned_model_uses_shared_loader_for_auto_flow(self):
        """Auto device mapping should flow through unchanged to the shared loader."""
        from script.evaluation.utils import load_fine_tuned_model

        with mock.patch(
            "script.evaluation.utils.load_model_and_tokenizer",
            return_value=(object(), object()),
        ) as loader_mock:
            load_fine_tuned_model(
                "hf-model",
                base_model_name="base-model",
                device_map_auto=True,
                quantize_4bit=True,
                as_trainable=False,
            )

        loader_mock.assert_called_once_with(
            "hf-model",
            device_map="auto",
            use_fp16=True,
            ddp_mode=False,
            quantize_4bit=True,
            lora=None,
            base_model_name_for_adapters="base-model",
            adapter_trainable=False,
        )

    def test_auto_accelerator_without_cuda_falls_back_to_cpu(self):
        """Auto accelerator should resolve to CPU when CUDA is unavailable."""
        from src.training.device_map import resolve_training_device_map

        resolved = resolve_training_device_map(
            requested_device_map="auto",
            accelerator="auto",
            devices="auto",
            cuda_available=False,
            cuda_device_count=0,
        )

        self.assertEqual(resolved, "cpu")


class TestLazyEvaluationCallbacks(unittest.TestCase):
    """Test lazy dataset loading in evaluation callbacks."""

    @classmethod
    def setUpClass(cls) -> None:
        missing = missing_optional_packages("lightning")
        if missing:
            raise unittest.SkipTest(
                "Missing optional dependency(s): " + ", ".join(sorted(missing))
            )

    def test_muse_callback_loads_datasets_lazily(self):
        """MUSE callback should defer dataset materialization until evaluation runs."""
        from src.training.callback.evaluation import MUSEEvaluationCallback

        callback = MUSEEvaluationCallback(
            forget_path="/tmp/forget.json",
            retain_path="/tmp/retain.json",
            paraphrased_path="/tmp/paraphrased.json",
            non_training_path="/tmp/non_training.json",
            output_dir="/tmp/muse_callback_test",
        )

        with mock.patch(
            "src.training.callback.evaluation.load_evaluation_data",
            return_value=([{"question": "q", "answer": "a"}], [{"question": "r", "answer": "b"}], None, None),
        ) as load_mock:
            load_mock.assert_not_called()
            callback._ensure_examples_loaded()
            load_mock.assert_called_once_with(
                "/tmp/forget.json",
                "/tmp/retain.json",
                "/tmp/paraphrased.json",
                "/tmp/non_training.json",
            )
            callback._ensure_examples_loaded()
            load_mock.assert_called_once()

    def test_raquel_callback_loads_datasets_lazily(self):
        """RAQUEL callback should defer dataset materialization until evaluation runs."""
        from src.training.callback.evaluation import RAQUELEvaluationCallback

        callback = RAQUELEvaluationCallback(
            affected_path="/tmp/affected.json",
            unaffected_path="/tmp/unaffected.json",
            output_dir="/tmp/raquel_callback_test",
        )

        with mock.patch(
            "src.training.callback.evaluation.load_raquel_examples",
            side_effect=[
                [{"question": "a", "answer": "1"}],
                [{"question": "b", "answer": "2"}],
            ],
        ) as load_mock:
            load_mock.assert_not_called()
            callback._ensure_examples_loaded()
            self.assertEqual(load_mock.call_count, 2)
            callback._ensure_examples_loaded()
            self.assertEqual(load_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
