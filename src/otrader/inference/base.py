"""Provider-neutral inference interface.

A backend runs ONE agentic episode: system prompt + user prompt + tools -> final JSON text.
It owns the tool-call loop (or delegates it to the Cursor agent), returns usage and a tool trace.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

ToolFn = Callable[[dict], dict]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_inr: float | None = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cost_inr = (self.cost_inr + other.cost_inr
                         if self.cost_inr is not None and other.cost_inr is not None else None)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class EpisodeResult:
    text: str
    usage: Usage
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model_id: str = ""
    raw: Any = None


class Backend:
    name = "base"

    def run_episode(self, system: str, user: str, tools: dict[str, ToolFn], tool_specs: dict[str, dict],
                    model_id: str, max_tool_turns: int = 8) -> EpisodeResult:
        raise NotImplementedError


def extract_json(text: str) -> dict:
    """Pull the first top-level JSON object out of model text (tolerates ```json fences)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError("no JSON object found in model output")


# Rough INR cost table (USD per 1M tokens) for the ledger only; the plan's flat fee is what you pay.
USD_INR = 88.0
PRICES_USD_PER_M = {
    "grok-4.6": (2.0, 6.0), "grok-4.5": (2.0, 6.0), "composer-2.5": (0.5, 2.5),
    "claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0), "claude-sonnet-4.6": (3.0, 15.0),
    "claude-haiku-4.5": (1.0, 5.0),
    "gpt-5.6-luna": (0.2, 1.2), "gpt-5.6-terra": (2.0, 12.0), "gpt-5.6-sol": (4.0, 20.0),
    "gemini-3.8-flash": (0.75, 3.5), "gemini-3.1-pro": (2.0, 12.0),
    "deepseek-v4-flash": (0.14, 0.28), "deepseek-v4-pro": (1.74, 3.48),
}


def estimate_cost_inr(model_id: str, inp: int, out: int) -> float:
    key = next((k for k in PRICES_USD_PER_M if k in model_id.lower()), None)
    if not key:
        return 0.0
    pi, po = PRICES_USD_PER_M[key]
    return round((inp * pi + out * po) / 1e6 * USD_INR, 4)
