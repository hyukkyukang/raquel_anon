import json

import pytest

from metrics.accuracy import (
    SemanticAccuracyMetric,
    SemanticMetricConfig,
)


class _StubLLMCaller:
    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    def __call__(self, _prompt, **_kwargs):
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


def test_semantic_metric_returns_accuracy():
    payload = {
        "results": [
            {"index": 0, "match": True, "reason": "same fact"},
            {"index": 1, "match": False, "reason": "different detail"},
        ]
    }
    stub_llm = _StubLLMCaller(responses=[json.dumps(payload)])

    metric = SemanticAccuracyMetric(
        SemanticMetricConfig(batch_size=4, max_retries=1),
        llm_caller=stub_llm,
    )
    metric.update(
        ["What is alpha?", "What is beta?"],
        ["Answer A", "Answer B"],
        ["Answer A", "Different"],
    )

    accuracy = metric.compute()

    assert accuracy == pytest.approx(0.5)
    assert stub_llm.calls == 1


def test_semantic_metric_uses_cache():
    payload = {
        "results": [
            {"index": 0, "match": True, "reason": "cached"},
        ]
    }
    stub_llm = _StubLLMCaller(responses=[json.dumps(payload)])

    metric = SemanticAccuracyMetric(
        SemanticMetricConfig(batch_size=1, max_retries=1),
        llm_caller=stub_llm,
    )

    metric.update(["What is alpha?"], ["Answer"], ["Answer"])
    assert metric.compute() == pytest.approx(1.0)
    metric.reset()

    metric.update(["What is alpha?"], ["Answer"], ["Answer"])
    assert metric.compute() == pytest.approx(1.0)

    # Cache should avoid second API call.
    assert stub_llm.calls == 1


def test_semantic_metric_exposes_last_results():
    payload = {
        "results": [
            {"index": 0, "match": True, "reason": "aligned"},
            {"index": 1, "match": False, "reason": "mismatch"},
        ]
    }
    stub_llm = _StubLLMCaller(responses=[json.dumps(payload)])

    metric = SemanticAccuracyMetric(
        SemanticMetricConfig(batch_size=4, max_retries=1),
        llm_caller=stub_llm,
    )
    metric.update(
        ["Question 1", "Question 2"],
        ["A", "B"],
        ["A", "C"],
    )

    metric.compute()
    assert metric.get_last_eval_results() == [
        ("Question 1", "A", "A", True),
        ("Question 2", "B", "C", False),
    ]

    metric.reset()
    assert metric.get_last_eval_results() == []
