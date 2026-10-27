"""Immutable risk rules. The agent cannot see or change this module's parameters at runtime."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskRules:
    max_positions: int
    max_position_frac: float
    max_gross_exposure_frac: float
    no_shorting: bool
    max_orders_per_day: int

    @classmethod
    def from_config(cls, cfg: dict) -> "RiskRules":
        return cls(**cfg)

    def describe(self) -> str:
        return (f"Long-only={self.no_shorting}; max {self.max_positions} positions; "
                f"max {self.max_position_frac:.0%} of equity per position; "
                f"max gross exposure {self.max_gross_exposure_frac:.0%}; "
                f"max {self.max_orders_per_day} orders/day. These are enforced by the engine.")
