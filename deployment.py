"""Resolved deployment topology shared between placement and calculation layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResolvedAssignment:
    """One assignment with its GPU and phase-specific topology resolved."""

    assignment: Any
    gpu_spec: Any
    phase: str
    prefill_mem: Any = None
    decode_mem: Any = None

    def __getattr__(self, name: str) -> Any:
        if self.phase == "prefill":
            if name == "tp":
                return self.assignment.prefill_tp
            if name == "pp":
                return self.assignment.prefill_pp
            if name == "dp":
                return self.assignment.prefill_dp
        return getattr(self.assignment, name)


@dataclass(frozen=True)
class Deployment:
    """All runnable assignments resolved for both prefill and decode phases."""

    prefill: tuple[ResolvedAssignment, ...]
    decode: tuple[ResolvedAssignment, ...]

    def for_phase(self, phase: str = "decode") -> tuple[ResolvedAssignment, ...]:
        return self.prefill if phase == "prefill" else self.decode
