"""
asas/core/state.py
==================
SystemState: the fundamental data abstraction of ASAS.

Theoretical basis
-----------------
ASAS separates two logically distinct state layers per sector:

    Systemic layer  (risk r_i):       what is happening in the world
    Epistemic layer (confidence c_i): what the system knows about it

This separation reflects the core claim of ASAS:
    Urban failures arise not from high systemic risk alone, but from
    the gap between systemic risk and epistemic confidence.
    The Frankfurt Marathon incident: all data existed; no system
    integrated it into situational awareness.

Control variable:
    Allocation a_i ∈ [0,1], Σa_i = 1 — the ONLY variable directly controlled.

Derived:
    Uncertainty  u_i = 1 - c_i        (epistemic gap)
    Base score   s_i = r_i · u_i      (AND logic: risk AND unknown)
    H-contrib    h_i = a_i · r_i · u_i
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


@dataclass
class SectorState:
    """State of one sector: systemic risk + epistemic confidence + allocation."""
    sector_id:  str
    risk:       float = 0.3   # systemic layer
    confidence: float = 0.5   # epistemic layer
    allocation: float = 0.0   # control variable

    @property
    def uncertainty(self) -> float:
        return 1.0 - self.confidence

    @property
    def base_score(self) -> float:
        """r_i · u_i — core priority signal."""
        return self.risk * self.uncertainty

    @property
    def entropy_contribution(self) -> float:
        """a_i · r_i · u_i — contribution to H(t)."""
        return self.allocation * self.risk * self.uncertainty

    def __repr__(self) -> str:
        return (f"Sector({self.sector_id}: "
                f"r={self.risk:.2f} c={self.confidence:.2f} "
                f"a={self.allocation:.2f} score={self.base_score:.3f})")


@dataclass
class SystemState:
    """
    Complete ASAS network state at one instant.

    Primary data structure passed between all components:
    dynamics, objective, policy, cognitive hub.

    Attributes
    ----------
    sectors  : sector_id → SectorState
    coupling : (src, tgt) → spillover weight
    step     : simulation step counter
    """
    sectors:   Dict[str, SectorState]
    coupling:  Dict[Tuple[str, str], float]
    step:      int = 0
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    @property
    def sector_ids(self) -> List[str]:
        return list(self.sectors.keys())

    @property
    def N(self) -> int:
        return len(self.sectors)

    def risk_vector(self)       -> Dict[str, float]:
        return {sid: s.risk       for sid, s in self.sectors.items()}

    def confidence_vector(self) -> Dict[str, float]:
        return {sid: s.confidence for sid, s in self.sectors.items()}

    def allocation_vector(self) -> Dict[str, float]:
        return {sid: s.allocation for sid, s in self.sectors.items()}

    def score_vector(self)      -> Dict[str, float]:
        return {sid: s.base_score for sid, s in self.sectors.items()}

    def outgoing(self, sid: str) -> List[Tuple[str, float]]:
        return [(t, w) for (s, t), w in self.coupling.items() if s == sid]

    def incoming(self, sid: str) -> List[Tuple[str, float]]:
        return [(s, w) for (s, t), w in self.coupling.items() if t == sid]

    def is_source(self, sid: str) -> bool:
        return any(s == sid for s, _ in self.coupling)

    def is_sink(self, sid: str) -> bool:
        return (any(t == sid for _, t in self.coupling)
                and not self.is_source(sid))

    def to_dict(self) -> Dict:
        """Structured export for logging / LLM consumption."""
        return {
            "step": self.step,
            "sectors": {
                sid: {
                    "risk":        round(s.risk, 4),
                    "confidence":  round(s.confidence, 4),
                    "uncertainty": round(s.uncertainty, 4),
                    "allocation":  round(s.allocation, 4),
                    "base_score":  round(s.base_score, 4),
                    "H_contrib":   round(s.entropy_contribution, 4),
                    "is_source":   self.is_source(sid),
                    "is_sink":     self.is_sink(sid),
                }
                for sid, s in self.sectors.items()
            },
        }

    def __repr__(self) -> str:
        lines = [f"SystemState(step={self.step}, N={self.N})"]
        for s in self.sectors.values():
            lines.append(f"  {s}")
        return "\n".join(lines)


def make_state(
    sectors: Dict[str, Dict],
    coupling: Dict[Tuple[str, str], float],
    step: int = 0,
) -> SystemState:
    """
    Convenience constructor.

    Example
    -------
    state = make_state(
        sectors={
            "Hbf":   {"risk": 0.9, "confidence": 0.7},
            "Messe": {"risk": 0.4, "confidence": 0.05},
        },
        coupling={("Messe", "Hbf"): 0.7},
    )
    """
    n = len(sectors)
    return SystemState(
        sectors={
            sid: SectorState(
                sector_id=sid,
                risk=cfg.get("risk", 0.3),
                confidence=cfg.get("confidence", 0.5),
                allocation=cfg.get("allocation", 1.0 / n),
            )
            for sid, cfg in sectors.items()
        },
        coupling=coupling,
        step=step,
    )
