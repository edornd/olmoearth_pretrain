"""Trainer based on olmo_core."""

from logging import getLogger
from typing import Any

import torch
from olmo_core.train.trainer import Trainer

logger = getLogger(__name__)


class HeliosTrainer(Trainer):
    """Trainer for Helios."""

    def model_forward(self, micro_batch: dict[str, Any]) -> torch.Tensor:
        """Run a forward pass on a micro-batch, returning the logits."""
        pass

    def get_losses(
        self,
    ) -> dict[str, Any]:
        """Compute the losses for a micro-batch and logits."""
        pass

    def eval_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a batch."""
        pass

    def train_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Train a batch."""
        pass

    def _fit_epoch(self):
        pass

    def _dry_run_batch(self):
        logger.debug("dry run batch")
        return {}
