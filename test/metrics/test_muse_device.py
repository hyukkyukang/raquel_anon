"""Tests for shared MUSE device resolution behavior."""

from __future__ import annotations

import unittest
from unittest import mock

from test.training._deps import missing_optional_packages

if missing_optional_packages("torch"):
    torch = None
else:
    import torch


if torch is not None:

    class _SingleParamModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))


    class _NoParamModel(torch.nn.Module):
        pass
else:

    class _SingleParamModel:
        pass

    class _NoParamModel:
        pass


class TestMuseDeviceResolution(unittest.TestCase):
    """Regression tests for model-aware MUSE device resolution."""

    @classmethod
    def setUpClass(cls) -> None:
        missing = missing_optional_packages("torch")
        if missing:
            raise unittest.SkipTest(
                "Missing optional dependency(s): " + ", ".join(sorted(missing))
            )

    def test_resolve_generation_device_prefers_hf_device_map(self):
        """HF auto-mapped models should resolve to their mapped device, not the first parameter."""
        from src.evaluation import resolve_generation_device

        model = _SingleParamModel()
        model.hf_device_map = {"model.embed_tokens": 1}

        resolved = resolve_generation_device(model, None)

        self.assertEqual(str(resolved), "cuda:1")

    def test_muse_evaluator_uses_model_device_over_requested_device(self):
        """MUSE evaluation should follow the model's actual device placement."""
        from src.evaluation.muse import MUSEEvaluator

        model = _SingleParamModel()

        with self.assertLogs("src.evaluation.raquel", level="WARNING") as logs:
            evaluator = MUSEEvaluator(model, tokenizer=object(), device="cuda")

        self.assertEqual(str(evaluator.device), "cpu")
        self.assertTrue(any("using model device" in message for message in logs.output))

    def test_no_parameter_model_falls_back_to_cpu_when_cuda_unavailable(self):
        """Parameterless models should still resolve to a safe fallback device."""
        from src.evaluation import resolve_generation_device

        with mock.patch("src.evaluation.raquel.torch.cuda.is_available", return_value=False):
            resolved = resolve_generation_device(_NoParamModel(), None)

        self.assertEqual(str(resolved), "cpu")


if __name__ == "__main__":
    unittest.main()
