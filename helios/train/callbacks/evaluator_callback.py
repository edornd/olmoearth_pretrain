"""Downstream evaluator callback."""

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from olmo_core.eval.evaluator import Evaluator
from olmo_core.train.callbacks.callback import Callback, CallbackConfig
from olmo_core.train.callbacks.evaluator_callback import EvaluatorCallback
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from torch.utils.data import DataLoader
from upath import UPath

from helios.evals.datasets import GeobenchDataset
from helios.evals.embeddings import get_embeddings
from helios.evals.knn import run_knn

logger = logging.getLogger(__name__)


class DownstreamEvaluator(Evaluator):
    """Evaluator for downstream tasks."""

    metric_type_to_label = {
        "f1": "F1 score",
    }

    def __init__(
        self,
        *,
        name: str,
        task: str,
        trainer: Trainer,
        device: torch.device | None = None,
        dp_process_group: dist.ProcessGroup | None = None,
    ) -> None:
        """Initialize the downstream evaluator."""
        geobench_dir = UPath("/weka/dfive-default/presto-geobench/dataset/geobench")

        train_ds = GeobenchDataset(geobench_dir, "m-eurosat", "train", "default")
        train_loader = DataLoader(train_ds, collate_fn=GeobenchDataset.collate_fn)
        val_loader = DataLoader(
            GeobenchDataset(geobench_dir, "m-eurosat", "valid", "default"),
            collate_fn=GeobenchDataset.collate_fn,
        )
        train_embeddings, train_labels = get_embeddings(
            data_loader=train_loader,
            model=trainer.train_module.model.target_encoder,
            patch_size=trainer.train_module.model.encoder.max_patch_size,
        )
        val_embeddings, test_labels = get_embeddings(
            data_loader=val_loader,
            model=trainer.train_module.model.target_encoder,
            patch_size=trainer.train_module.model.encoder.max_patch_size,
        )
        val_result = run_knn(
            eval_type="KNN-20",
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=val_embeddings,
            test_labels=test_labels,
            num_classes=train_ds.num_classes,
            is_multilabel=train_ds.is_multilabel,
            device=device,
        )
        logger.info(val_result)

    def update_metrics(
        self,
        batch: dict[str, Any],
        ce_loss: torch.Tensor | None,
        logits: torch.Tensor | None,
    ) -> None:
        """Update the metrics."""
        del ce_loss
        self.metric.update(batch, logits)

    def compute_metrics(self) -> dict[str, torch.Tensor]:
        """Compute the metrics."""
        metric_type_to_value = self.metric.compute()
        outputs = {}
        for metric_type, value in metric_type_to_value.items():
            key = f"{self.label} ({self.metric_type_to_label[metric_type]})"
            outputs[key] = value
        return outputs

    def reset_metrics(self) -> None:
        """Reset the metrics."""
        self.metric.reset()


@dataclass
class DownstreamEvaluatorCallbackConfig(CallbackConfig):
    """Config for the downstream evaluator callback."""

    tasks: list[str]
    eval_interval: int = 10
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(10))
    log_interval: int = 5
    enabled: bool = True

    def build(self, trainer: "Trainer") -> Callback | None:
        """Build the downstream evaluator callback."""
        if not self.enabled:
            return None

        evaluators: list[Evaluator] = []
        for task in self.tasks:
            evaluators.append(
                DownstreamEvaluator(
                    name="downstream",
                    task=task,
                    trainer=trainer,
                    device=trainer.device,
                    dp_process_group=trainer.dp_process_group,
                )
            )

        return EvaluatorCallback(
            evaluators=evaluators,
            eval_interval=self.eval_interval,
            log_interval=self.log_interval,
            eval_duration=self.eval_duration,
        )
