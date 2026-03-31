"""Custom progress bar callback."""

import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar


class CustomProgressBar(TQDMProgressBar):
    """
    Custom progress bar that displays loss and learning rate.

    This extends the default TQDMProgressBar to show additional metrics
    in a clean format during training.
    """

    def __init__(self, refresh_rate: int = 1):
        """
        Initialize custom progress bar.

        Args:
            refresh_rate: How often to refresh the progress bar (in steps)
        """
        super().__init__(refresh_rate=refresh_rate)

    def get_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> dict:
        """
        Get metrics to display in progress bar.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module

        Returns:
            Dictionary of metrics to display
        """
        # Get base metrics from parent
        items = super().get_metrics(trainer, pl_module)

        # Remove version number
        items.pop("v_num", None)

        # Format loss values to 4 decimal places
        for key in list(items.keys()):
            if "loss" in key:
                items[key] = f"{items[key]:.4f}"

        return items
