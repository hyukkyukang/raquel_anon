"""Unit tests for verifying all training module imports."""

import unittest
from unittest import mock

from test.training._deps import require_optional_packages


class TestTrainingImports(unittest.TestCase):
    """Test that all refactored training modules can be imported."""

    def test_lightweight_data_imports(self):
        """Test lightweight data helpers that should import without ML runtimes."""
        from src.training.data import (
            create_idk_dataset,
            format_examples,
            get_idk_response,
            load_dataset,
        )

        self.assertTrue(callable(format_examples))
        self.assertTrue(callable(get_idk_response))
        self.assertTrue(callable(create_idk_dataset))
        self.assertTrue(callable(load_dataset))

    def test_heavy_data_imports(self):
        """Test data imports that depend on torch/transformers."""
        require_optional_packages(self, "torch", "transformers")

        from src.training.data import CustomDataCollator, tokenize_function

        self.assertTrue(callable(tokenize_function))
        self.assertTrue(callable(CustomDataCollator))

    def test_datamodule_imports(self):
        """Test LightningDataModule imports."""
        require_optional_packages(
            self, "datasets", "lightning", "omegaconf", "torch", "transformers"
        )

        from src.training.data.pl_module import (
            BaseDataModule,
            FineTuneDataModule,
            UnlearningDataModule,
        )

        # Verify all are classes
        self.assertTrue(isinstance(BaseDataModule, type))
        self.assertTrue(isinstance(FineTuneDataModule, type))
        self.assertTrue(isinstance(UnlearningDataModule, type))

        # Verify inheritance
        self.assertTrue(issubclass(FineTuneDataModule, BaseDataModule))
        self.assertTrue(issubclass(UnlearningDataModule, BaseDataModule))

    def test_loss_imports(self):
        """Test loss function imports."""
        require_optional_packages(self, "torch", "transformers")

        from src.training.loss import (
            DPOLoss,
            GradDescentLoss,
            GradientAscentLoss,
            IDKLoss,
            KLDivergenceLoss,
            NPOLoss,
        )

        # Verify all are classes
        self.assertTrue(isinstance(GradientAscentLoss, type))
        self.assertTrue(isinstance(NPOLoss, type))
        self.assertTrue(isinstance(IDKLoss, type))
        self.assertTrue(isinstance(DPOLoss, type))
        self.assertTrue(isinstance(GradDescentLoss, type))
        self.assertTrue(isinstance(KLDivergenceLoss, type))

    def test_model_imports(self):
        """Test Lightning module imports."""
        require_optional_packages(
            self,
            "lightning",
            "torch",
            "torchmetrics",
            "transformers",
        )

        from src.training.model import (
            BaseLightningModule,
            FinetuneModule,
            UnlearningModule,
        )

        # Verify all are classes
        self.assertTrue(isinstance(BaseLightningModule, type))
        self.assertTrue(isinstance(FinetuneModule, type))
        self.assertTrue(isinstance(UnlearningModule, type))

        # Verify inheritance
        self.assertTrue(issubclass(FinetuneModule, BaseLightningModule))
        self.assertTrue(issubclass(UnlearningModule, BaseLightningModule))

    def test_callback_imports(self):
        """Test callback imports."""
        require_optional_packages(
            self,
            "lightning",
            "numpy",
            "rouge_score",
            "torch",
            "torchmetrics",
            "tqdm",
            "transformers",
        )

        from src.training.callback import CustomProgressBar, MUSEEvaluationCallback

        # Verify all are classes
        self.assertTrue(isinstance(CustomProgressBar, type))
        self.assertTrue(isinstance(MUSEEvaluationCallback, type))

    def test_lightweight_training_exports(self):
        """Test lightweight training metadata and path helpers."""
        from src.training import (
            METHOD_DESCRIPTIONS,
            UNLEARNING_METHODS,
            check_model_exists,
            get_base_model_dir_component,
            needs_idk_dataset,
            needs_reference_model,
            parse_unlearning_method,
        )

        self.assertTrue(callable(get_base_model_dir_component))
        self.assertTrue(callable(check_model_exists))
        self.assertTrue(callable(parse_unlearning_method))
        self.assertTrue(callable(needs_idk_dataset))
        self.assertTrue(callable(needs_reference_model))

        # Verify constants
        self.assertIsInstance(UNLEARNING_METHODS, list)
        self.assertIsInstance(METHOD_DESCRIPTIONS, dict)
        self.assertEqual(len(UNLEARNING_METHODS), 8)
        self.assertEqual(len(METHOD_DESCRIPTIONS), 8)

    def test_heavy_utils_imports(self):
        """Test heavy training utilities that require the model stack."""
        require_optional_packages(self, "lightning", "peft", "torch", "transformers")

        from src.training.utils import load_model_and_tokenizer

        self.assertTrue(callable(load_model_and_tokenizer))

    def test_evaluation_imports(self):
        """Test evaluation module imports."""
        require_optional_packages(
            self, "numpy", "rouge_score", "torch", "tqdm", "transformers"
        )

        from src.evaluation import MUSEEvaluator

        # Verify it's a class
        self.assertTrue(isinstance(MUSEEvaluator, type))

    def test_main_training_module(self):
        """Test main training module import."""
        import src.training

        # Verify module has expected attributes
        self.assertTrue(hasattr(src.training, "datamodules"))
        self.assertTrue(hasattr(src.training, "model"))
        self.assertTrue(hasattr(src.training, "loss"))
        self.assertTrue(hasattr(src.training, "callback"))
        self.assertTrue(hasattr(src.training, "logger"))
        self.assertTrue(hasattr(src.training, "data"))


class TestLegacySurface(unittest.TestCase):
    """Test that legacy surfaces remain available but are no longer advertised."""

    def test_generator_all_hides_legacy_schema_generator(self):
        """The supported generator export surface should omit legacy entries."""
        import src.generator

        self.assertNotIn("SchemaGenerator", src.generator.__all__)

    def test_generator_legacy_schema_generator_warns(self):
        """Accessing the legacy generator export should emit a deprecation warning."""
        import src.generator

        dummy_module = type("DummyModule", (), {"SchemaGenerator": object()})()
        with mock.patch("src.generator.import_module", return_value=dummy_module):
            with self.assertWarns(DeprecationWarning):
                value = getattr(src.generator, "SchemaGenerator")

        self.assertIs(value, dummy_module.SchemaGenerator)


class TestUnlearningMethods(unittest.TestCase):
    """Test unlearning method utilities."""

    def test_unlearning_methods_list(self):
        """Test that all 8 unlearning methods are defined."""
        from src.training.methods import UNLEARNING_METHODS

        expected_methods = [
            "ga_gd",
            "ga_kl",
            "npo_gd",
            "npo_kl",
            "idk_gd",
            "idk_kl",
            "dpo_gd",
            "dpo_kl",
        ]

        self.assertEqual(set(UNLEARNING_METHODS), set(expected_methods))

    def test_method_descriptions(self):
        """Test that all methods have descriptions."""
        from src.training.methods import METHOD_DESCRIPTIONS, UNLEARNING_METHODS

        for method in UNLEARNING_METHODS:
            self.assertIn(method, METHOD_DESCRIPTIONS)
            self.assertIsInstance(METHOD_DESCRIPTIONS[method], str)
            self.assertGreater(len(METHOD_DESCRIPTIONS[method]), 0)

    def test_parse_unlearning_method(self):
        """Test parsing of unlearning method strings."""
        from src.training.methods import parse_unlearning_method

        # Test valid methods
        self.assertEqual(parse_unlearning_method("ga_gd"), ("ga", "gd"))
        self.assertEqual(parse_unlearning_method("ga_kl"), ("ga", "kl"))
        self.assertEqual(parse_unlearning_method("npo_gd"), ("npo", "gd"))
        self.assertEqual(parse_unlearning_method("npo_kl"), ("npo", "kl"))
        self.assertEqual(parse_unlearning_method("idk_gd"), ("idk", "gd"))
        self.assertEqual(parse_unlearning_method("idk_kl"), ("idk", "kl"))
        self.assertEqual(parse_unlearning_method("dpo_gd"), ("dpo", "gd"))
        self.assertEqual(parse_unlearning_method("dpo_kl"), ("dpo", "kl"))

        # Test invalid method
        with self.assertRaises(ValueError):
            parse_unlearning_method("invalid_method")

    def test_needs_idk_dataset(self):
        """Test detection of methods requiring IDK dataset."""
        from src.training import needs_idk_dataset

        # Methods that need IDK dataset
        self.assertTrue(needs_idk_dataset("idk_gd"))
        self.assertTrue(needs_idk_dataset("idk_kl"))
        self.assertTrue(needs_idk_dataset("dpo_gd"))
        self.assertTrue(needs_idk_dataset("dpo_kl"))

        # Methods that don't need IDK dataset
        self.assertFalse(needs_idk_dataset("ga_gd"))
        self.assertFalse(needs_idk_dataset("ga_kl"))
        self.assertFalse(needs_idk_dataset("npo_gd"))
        self.assertFalse(needs_idk_dataset("npo_kl"))

    def test_needs_reference_model(self):
        """Test detection of methods requiring reference model."""
        from src.training import needs_reference_model

        # Methods that need reference model (KL or DPO)
        self.assertTrue(needs_reference_model("ga_kl"))
        self.assertTrue(needs_reference_model("npo_gd"))
        self.assertTrue(needs_reference_model("npo_kl"))
        self.assertTrue(needs_reference_model("idk_kl"))
        self.assertTrue(needs_reference_model("dpo_kl"))
        self.assertTrue(needs_reference_model("dpo_gd"))  # DPO always needs reference

        # Methods that don't need reference model
        self.assertFalse(needs_reference_model("ga_gd"))
        self.assertFalse(needs_reference_model("idk_gd"))


class TestDataTransforms(unittest.TestCase):
    """Test data transformation functions."""

    def test_format_examples(self):
        """Test example formatting."""
        from src.training.data import format_examples

        examples = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is the capital of France?", "answer": "Paris"},
        ]

        formatted = format_examples(examples)

        self.assertEqual(len(formatted), 2)
        self.assertIn("text", formatted[0])
        self.assertIn("What is 2+2?", formatted[0]["text"])
        self.assertIn("4", formatted[0]["text"])
        self.assertIn("Question:", formatted[0]["text"])
        self.assertIn("Answer:", formatted[0]["text"])

    def test_get_idk_response(self):
        """Test IDK response generation."""
        from src.training.data import get_idk_response

        # Test 'first' variation
        response = get_idk_response("first")
        self.assertEqual(response, "I don't know.")

        # Test 'random' variation (should be one of the templates)
        response = get_idk_response("random")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

        # Test default fallback
        response = get_idk_response("invalid")
        self.assertEqual(response, "I don't know.")

    def test_create_idk_dataset(self):
        """Test IDK dataset creation."""
        from src.training.data import create_idk_dataset

        forget_examples = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is the capital of France?", "answer": "Paris"},
        ]

        idk_examples = create_idk_dataset(forget_examples, idk_variation="first")

        self.assertEqual(len(idk_examples), 2)
        self.assertEqual(idk_examples[0]["question"], "What is 2+2?")
        self.assertEqual(idk_examples[0]["answer"], "I don't know.")
        self.assertEqual(idk_examples[1]["question"], "What is the capital of France?")
        self.assertEqual(idk_examples[1]["answer"], "I don't know.")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_get_base_model_dir_component(self):
        """Test base model directory name generation."""
        from src.training import get_base_model_dir_component

        # Test known mappings
        self.assertEqual(
            get_base_model_dir_component("meta-llama/Llama-3.2-1B"), "llama-3.2-1b"
        )
        self.assertEqual(
            get_base_model_dir_component("meta-llama/Llama-3.2-1B-Instruct"),
            "llama-3.2-1b",
        )

        # Test sanitization
        result = get_base_model_dir_component("some/unknown/model")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Should not have consecutive dashes
        self.assertNotIn("--", result)


if __name__ == "__main__":
    unittest.main()
