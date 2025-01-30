"""Trainer based on olmo_core."""

from logging import getLogger
from typing import Any

import torch
from olmo_core.train.trainer import Trainer

logger = getLogger(__name__)



class HeliosTrainer(Trainer):
    """Trainer for Helios.

    Leaving this in for now so we can hack around any rough edges but ideally Trainer should be agnostic"""
    pass