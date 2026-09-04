"""Configuration for the standalone Soft Penalty v2 objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class RiskConfig:
    """Fully resolved risk objective and CEM calibration parameters."""

    mode: str = "goal_cvar_eot"
    alpha: float = 1.0
    num_samples: int = 300
    topk: int = 30
    tail_fraction: float = 0.4
    tail_count: Optional[int] = None
    cap: float = 2.0
    eps: float = 1e-8

    # Kept only for an apples-to-apples v1 control.
    lambda_weight: Optional[float] = None
    distance_scale: float = 1.0
    uncertainty_scale: float = 1.0

    MODES = frozenset(
        {
            "mc_terminal",
            "soft_v1_control",
            "goal_cvar_eot",
            "terminal_sigma_eot",
        }
    )

    def __post_init__(self) -> None:
        if self.mode not in self.MODES:
            raise ValueError(
                f"Unknown risk mode {self.mode!r}; expected one of {sorted(self.MODES)}"
            )
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if not 1 <= self.topk <= self.num_samples:
            raise ValueError("topk must be in [1, num_samples]")
        if not 0 < self.tail_fraction <= 1:
            raise ValueError("tail_fraction must be in (0, 1]")
        if self.tail_count is not None and self.tail_count < 1:
            raise ValueError("tail_count must be positive when specified")
        if self.cap < 0:
            raise ValueError("cap must be non-negative")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.lambda_weight is not None and self.lambda_weight < 0:
            raise ValueError("lambda_weight must be non-negative")
        if self.distance_scale <= 0 or self.uncertainty_scale <= 0:
            raise ValueError("v1 calibration scales must be positive")

    def resolved_tail_count(self, rollout_samples: int) -> int:
        """Resolve a valid upper-tail count (N=5 defaults to top two)."""
        if rollout_samples < 1:
            raise ValueError("rollout_samples must be positive")
        count = (
            self.tail_count
            if self.tail_count is not None
            else math.ceil(self.tail_fraction * rollout_samples)
        )
        return min(rollout_samples, max(1, int(count)))

    @property
    def resolved_lambda(self) -> float:
        if self.lambda_weight is not None:
            return float(self.lambda_weight)
        return float(self.alpha * self.distance_scale / self.uncertainty_scale)

    @classmethod
    def from_mapping(cls, values: Mapping) -> "RiskConfig":
        data = dict(values)
        if "lambda" in data:
            if "lambda_weight" in data:
                raise ValueError("Specify only one of lambda and lambda_weight")
            data["lambda_weight"] = data.pop("lambda")
        return cls(**data)
