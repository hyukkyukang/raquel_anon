"""Utilities for computing ROUGE metrics with torchmetrics."""

from typing import Dict, Iterable, Sequence

from torch import nn
from torchmetrics.text.rouge import ROUGEScore


class RougeMetric(nn.Module):
    """
    Thin wrapper around :class:`torchmetrics.text.ROUGEScore`.

    Provides a simple interface that:
        - accumulates predictions and targets across steps,
        - exposes scalar floats for logging, and
        - keeps metric state resettable between evaluation epochs.
    """

    def __init__(
        self,
        rouge_keys: Sequence[str] = ("rouge1", "rouge2", "rougeL"),
        use_stemmer: bool = True,
    ) -> None:
        super().__init__()
        self._metric = ROUGEScore(
            rouge_keys=tuple(rouge_keys),
            use_stemmer=use_stemmer,
        )
        self._has_updated = False

    def update(
        self,
        predictions: Iterable[str],
        references: Iterable[str],
    ) -> None:
        """
        Update accumulated statistics.

        Args:
            predictions: Iterable of generated answer strings.
            references: Iterable of reference answer strings.
        """
        preds_list = list(predictions)
        refs_list = list(references)

        if not preds_list:
            return

        if len(preds_list) != len(refs_list):
            raise ValueError(
                "Predictions and references must have the same length for ROUGE computation."
            )

        self._metric.update(preds=preds_list, target=refs_list)
        self._has_updated = True

    def compute(self) -> Dict[str, float]:
        """
        Compute ROUGE metrics accumulated so far.

        Returns:
            Dictionary of metric name to scalar float.
        """
        if not self._has_updated:
            return {}
        scores = self._metric.compute()
        return {name: float(value) for name, value in scores.items()}

    def compute_recall(self) -> Dict[str, float]:
        """Return only the recall components from the accumulated scores."""
        scores = self.compute()
        return {
            name: value for name, value in scores.items() if name.endswith("_recall")
        }

    def reset(self) -> None:
        """Reset accumulated state."""
        self._metric.reset()
        self._has_updated = False
