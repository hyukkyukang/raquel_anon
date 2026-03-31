"""Unit tests for data processing components."""

import os
import tempfile
import unittest

from test.training._deps import require_optional_packages


class DummyTokenizer:
    """Minimal tokenizer stub for unit tests."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._vocab = {
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
        }

    def _encode(self, text: str, max_length: int | None = None) -> list[int]:
        token_ids: list[int] = []
        cursor = 0
        while cursor < len(text):
            if text.startswith(self.eos_token, cursor):
                token_ids.append(self.eos_token_id)
                cursor += len(self.eos_token)
                continue

            char = text[cursor]
            token_ids.append(self._vocab.setdefault(char, len(self._vocab)))
            cursor += 1

        if max_length is not None:
            token_ids = token_ids[:max_length]
        return token_ids

    def __call__(
        self,
        texts,
        truncation: bool = True,
        max_length: int | None = None,
        add_special_tokens: bool = False,
        return_tensors=None,
        padding: bool = False,
        return_attention_mask: bool = False,
    ):
        del add_special_tokens, padding, return_tensors

        effective_max_length = max_length if truncation else None

        if isinstance(texts, list):
            input_ids = [
                self._encode(text, max_length=effective_max_length) for text in texts
            ]
            attention_mask = [[1] * len(ids) for ids in input_ids]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        input_ids = self._encode(str(texts), max_length=effective_max_length)
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class TestCustomDataCollator(unittest.TestCase):
    """Test CustomDataCollator."""

    def setUp(self):
        """Set up test fixtures."""
        require_optional_packages(self, "torch", "transformers")
        self.tokenizer = DummyTokenizer()

    def test_collator_initialization(self):
        """Test collator can be initialized."""
        from src.training.data import CustomDataCollator

        collator = CustomDataCollator(tokenizer=self.tokenizer, mlm=False)
        self.assertIsNotNone(collator)
        self.assertEqual(collator.tokenizer, self.tokenizer)

    def test_collator_preserves_labels(self):
        """Test that collator preserves pre-computed labels."""
        from src.training.data import CustomDataCollator

        collator = CustomDataCollator(tokenizer=self.tokenizer, mlm=False)

        # Create sample features with labels
        features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [-100, -100, 3, 4, 5],
            },
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, 3],
            },
        ]

        batch = collator(features)

        # Check that batch has required keys
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("labels", batch)

        # Check shapes
        self.assertEqual(batch["input_ids"].shape[0], 2)  # batch size
        self.assertEqual(batch["labels"].shape[0], 2)  # batch size

        # Check that padding was applied correctly
        max_len = max(len(f["input_ids"]) for f in features)
        self.assertEqual(batch["input_ids"].shape[1], max_len)


class TestDatasetLoading(unittest.TestCase):
    """Test dataset loading functionality."""

    def test_load_dataset_from_json(self):
        """Test loading dataset from JSON file."""
        import json

        from src.training.data import load_dataset

        # Create temporary JSON file
        test_data = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is the capital?", "answer": "Paris"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            # Load dataset
            dataset = load_dataset(temp_path)

            # Verify
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["question"], "What is 2+2?")
            self.assertEqual(dataset[0]["answer"], "4")
        finally:
            import os

            os.unlink(temp_path)


class TestTokenization(unittest.TestCase):
    """Test tokenization functions."""

    def setUp(self):
        """Set up test fixtures."""
        require_optional_packages(self, "transformers")
        self.tokenizer = DummyTokenizer()

    def test_tokenize_function_basic(self):
        """Test basic tokenization."""
        from src.training.data import tokenize_function

        examples = {
            "text": ["Question: What is 2+2?\nAnswer: 4"],
        }

        result = tokenize_function(examples, self.tokenizer)

        # Check required keys
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)

        # Check that we have tokenized output
        self.assertIsInstance(result["input_ids"], list)
        self.assertGreater(len(result["input_ids"]), 0)

    def test_tokenize_function_masks_prompt(self):
        """Test that prompt tokens are masked in labels."""
        from src.training.data import tokenize_function

        examples = {
            "text": ["Question: What is 2+2?\nAnswer: 4"],
        }

        result = tokenize_function(examples, self.tokenizer)

        # Labels should have -100 for prompt tokens
        labels = result["labels"][0]
        self.assertIn(-100, labels)

        # Not all labels should be -100 (answer should not be masked)
        self.assertTrue(any(label != -100 for label in labels))


class TestIDKDataset(unittest.TestCase):
    """Test IDK dataset creation."""

    def setUp(self):
        require_optional_packages(self, "transformers")

    def test_idk_templates_exist(self):
        """Test that IDK templates are defined."""
        from src.training.data.transforms import IDK_TEMPLATES

        self.assertIsInstance(IDK_TEMPLATES, list)
        self.assertGreater(len(IDK_TEMPLATES), 0)
        self.assertIn("I don't know.", IDK_TEMPLATES)

    def test_create_idk_dataset_preserves_questions(self):
        """Test that IDK dataset preserves original questions."""
        from src.training.data import create_idk_dataset

        forget_examples = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "Capital of France?", "answer": "Paris"},
        ]

        idk_dataset = create_idk_dataset(forget_examples, idk_variation="first")

        # Check length
        self.assertEqual(len(idk_dataset), len(forget_examples))

        # Check questions are preserved
        self.assertEqual(idk_dataset[0]["question"], forget_examples[0]["question"])
        self.assertEqual(idk_dataset[1]["question"], forget_examples[1]["question"])

        # Check answers are replaced with IDK
        self.assertEqual(idk_dataset[0]["answer"], "I don't know.")
        self.assertEqual(idk_dataset[1]["answer"], "I don't know.")

    def test_create_idk_dataset_random_variation(self):
        """Test IDK dataset with random variation."""
        from src.training.data import create_idk_dataset
        from src.training.data.transforms import IDK_TEMPLATES

        forget_examples = [
            {"question": "What is 2+2?", "answer": "4"},
        ]

        idk_dataset = create_idk_dataset(forget_examples, idk_variation="random")

        # Answer should be one of the IDK templates
        self.assertIn(idk_dataset[0]["answer"], IDK_TEMPLATES)


class TestUnlearningDataLoaderState(unittest.TestCase):
    """Test that the custom unlearning dataloader manages its own pickle-safe state."""

    def setUp(self):
        require_optional_packages(self, "torch")

    def test_unlearning_dataloader_strips_iterators_from_state(self):
        """The dataloader should not rely on a global DataLoader monkey patch."""
        import torch

        from src.training.data.dataloader import UnlearningDataLoader

        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self, values):
                self.values = values

            def __len__(self):
                return len(self.values)

            def __getitem__(self, idx):
                return {"sample_id": int(self.values[idx])}

        def collator(batch):
            return {
                "sample_id": torch.tensor(
                    [int(item["sample_id"]) for item in batch], dtype=torch.long
                )
            }

        loader = UnlearningDataLoader(
            forget_dataset=TinyDataset([0, 1, 2]),
            retain_dataset=TinyDataset([10, 11, 12]),
            collator=collator,
            train_batch_size=2,
            retain_batch_size=2,
            num_workers=0,
            multiprocessing_context=None,
        )
        loader._iterator = object()
        loader._retain_loader._iterator = object()

        state = loader.__getstate__()

        self.assertNotIn("_iterator", state)
        self.assertNotIn("_retain_loader", state)

        restored = object.__new__(UnlearningDataLoader)
        restored.__setstate__(state)
        self.assertTrue(hasattr(restored, "_retain_loader"))
        self.assertEqual(len(restored.retain_dataset), 3)


if __name__ == "__main__":
    unittest.main()
