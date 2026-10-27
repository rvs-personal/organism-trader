"""Strict historical clock. Only the runner advances it. Agents receive it read-only via the store."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Clock:
    today: int = 1            # simulation day index (1-based)
    _frozen: bool = field(default=False, repr=False)

    def advance(self) -> int:
        if self._frozen:
            raise RuntimeError("clock is frozen")
        self.today += 1
        return self.today

    def set(self, day: int) -> None:
        if self._frozen:
            raise RuntimeError("clock is frozen")
        self.today = int(day)

    def freeze(self) -> None:
        self._frozen = True
