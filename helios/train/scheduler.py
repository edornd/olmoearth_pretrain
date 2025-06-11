"""Schedulers."""

from dataclasses import dataclass

import torch
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim.scheduler import Scheduler, _linear_warmup


@dataclass
class PolyWithWarmup(Scheduler):
    """Polynomial learning rate schedule with a warmup."""

    warmup: int | None = None
    warmup_fraction: float | None = None
    power: float = 3.3219
    t_max: int | None = None
    warmup_min_lr: float = 0.0

    def __post_init__(self) -> None:
        """Set up warmups."""
        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError(
                "Either 'warmup_fraction' or 'warmup' must be specified."
            )

        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: float | torch.Tensor, current: int, t_max: int
    ) -> float | torch.Tensor:
        """Get learning rate."""
        t_max = t_max if self.t_max is None else self.t_max

        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup

        if current < warmup:
            return _linear_warmup(initial_lr, current, warmup, self.warmup_min_lr)
        elif current >= t_max:
            return 0
        else:
            current = current - warmup
            t_max = t_max - warmup
            return initial_lr * pow(1 - current / t_max, self.power)
