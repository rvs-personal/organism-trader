"""Structured outputs the model must produce. Validated with pydantic; one retry on failure."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, conlist


class OrderIntent(BaseModel):
    asset: str
    side: Literal["BUY", "SELL"]
    notional_inr: float | None = Field(None, description="INR to deploy (BUY). Engine converts to shares.")
    qty: int | None = Field(None, description="Shares to sell (SELL); omit to close the position.")
    reason: str = Field(..., max_length=300)


class Decision(BaseModel):
    """End-of-research output for one simulated day."""
    market_view: str = Field(..., max_length=600)
    orders: list[OrderIntent] = Field(default_factory=list, max_length=5)
    theses: dict[str, str] = Field(default_factory=dict, description="asset -> current thesis (short)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence that today's orders add value")
    watchlist: list[str] = Field(default_factory=list, max_length=8)
    note_to_self: str = Field("", max_length=400, description="Working memory carried to tomorrow")


class EpisodicEntry(BaseModel):
    sim_day: int
    summary: str = Field(..., max_length=500)
    lesson: str | None = Field(None, max_length=300)
    tags: list[str] = Field(default_factory=list, max_length=6)


class Reflection(BaseModel):
    """Every ~10 days."""
    what_worked: list[str] = Field(default_factory=list, max_length=6)
    what_failed: list[str] = Field(default_factory=list, max_length=6)
    belief_updates: dict[str, str] = Field(default_factory=dict, description="belief_key -> new statement")
    beliefs_to_drop: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list, max_length=6)
    perception_policy: str = Field(..., max_length=600, description="How I should research tomorrow")
    strategy_change: str | None = Field(None, max_length=1500, description="Full new strategy text, or null to keep")
    confidence_calibration_note: str = Field("", max_length=300)


class Review(Reflection):
    """Every ~50 days: same shape, longer horizon. Strategy change expected to be considered seriously."""
    long_term_memory_additions: list[str] = Field(default_factory=list, max_length=8)


class CandidateAssessment(BaseModel):
    asset: str
    disposition: Literal['act', 'hold', 'reject']
    reason: str = Field(..., max_length=240)


class InterventionDecision(Decision):
    candidate_assessments: list[CandidateAssessment] = Field(..., max_length=5,
        description='Strongest considered candidates and explicit blocking conditions; empty only if none is plausible')


class AllocationPlan(BaseModel):
    cash_reserve_inr: float = Field(..., ge=0, description='Cash you intend to retain after these orders; a plan, not a guaranteed fill amount')
    reserve_reason: str = Field(..., max_length=400, description='Why this much cash is useful now; cite risks or conditional future opportunities, never future knowledge')
    reserve_release_condition: str = Field(..., max_length=300, description='Observable condition for deploying the reserve, or explain why no reserve is needed')
    sizing_rationale: str = Field(..., max_length=500, description='Explain chosen position sizes, additions and diversification. No fixed starter size or required idle cash')


class AllocationDecision(InterventionDecision):
    orders: list['BuyAllocationOrder | SellAllocationOrder'] = Field(..., max_length=58, description='Executable orders for TODAY. Every planned purchase or sale must appear here. An empty list means no transactions and unchanged cash.')
    allocation_plan: AllocationPlan


class BuyAllocationOrder(BaseModel):
    asset: str
    side: Literal['BUY']
    notional_inr: float = Field(..., gt=0, description='Positive INR amount to spend on shares, before fees')
    qty: None = Field(None, description='BUY shares are calculated by the engine; always null')
    reason: str = Field(..., max_length=300)


class SellAllocationOrder(BaseModel):
    asset: str
    side: Literal['SELL']
    qty: int | None = Field(None, gt=0, description='Positive number of existing shares to sell, or null to close the position')
    notional_inr: None = None
    reason: str = Field(..., max_length=300)


AllocationDecision.model_rebuild()
