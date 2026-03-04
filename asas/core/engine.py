"""
asas/core/engine.py
===================
ASASEngine: control loop orchestrator.

Responsibilities
----------------
1. Hold current SystemState
2. Accept external signals from perception layer (ingest)
3. Call policy.allocate(state) → allocation
4. Call dynamics.step(state, allocation, signals) → new_state
5. Record history
6. Expose state to cognitive hub on demand

The engine contains NO mathematical formulas.
All math lives in dynamics.py, objective.py, and policy.py.
The engine only orchestrates calls between them.

Architecture
------------
                    ┌─────────────────┐
    external        │   ASASEngine    │
    signals ───────▶│                 │───▶ SystemState
                    │  policy         │
                    │  dynamics       │───▶ status_report
                    │  history        │         │
                    └─────────────────┘         ▼
                                          CognitiveHub
                                          (optional adapter)
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Callable, Tuple, Any

from .state import SystemState, SectorState, make_state
from .dynamics import step as dynamics_step
from .objective import system_entropy, status_report as make_status_report
from .policy import AllocationPolicy, SoftmaxPolicy


class ASASEngine:
    """
    Main ASAS control loop.

    Parameters
    ----------
    initial_state   : SystemState to begin from
    policy          : AllocationPolicy (default: SoftmaxPolicy)
    cognitive_hub   : optional CognitiveHub adapter
    dynamics_params : kwargs forwarded to dynamics.step()
                      (natural_decay, mitigation_strength, etc.)
    """

    def __init__(
        self,
        initial_state: SystemState,
        policy: Optional[AllocationPolicy] = None,
        cognitive_hub=None,
        **dynamics_params,
    ):
        self.state          = initial_state
        self.policy         = policy or SoftmaxPolicy()
        self.cognitive_hub  = cognitive_hub
        self.dynamics_params = dynamics_params

        self._history: List[SystemState] = []
        self._pending_signals: Dict[str, float] = {}

    # ── Signal ingestion ───────────────────────────────────────

    def ingest(
        self,
        signals: Dict[str, float],
        mode: str = "add",
    ) -> None:
        """
        Accept external risk signals from the perception layer.

        Parameters
        ----------
        signals : sector_id → risk delta
                  Positive = risk increase (e.g. strike event: {"Hbf": +0.4})
                  Negative = risk decrease (e.g. incident resolved)
        mode    : "add"     — accumulate until next step()
                  "replace" — discard previous pending signals
        """
        if mode == "replace":
            self._pending_signals = dict(signals)
        else:
            for sid, delta in signals.items():
                self._pending_signals[sid] = (
                    self._pending_signals.get(sid, 0.0) + delta
                )

    # ── Step execution ─────────────────────────────────────────

    def step(self) -> SystemState:
        """
        Advance the system by one time step.

        Order of operations:
            1. policy.allocate(state) → allocation
            2. dynamics.step(state, allocation, signals) → new_state
            3. Clear pending signals, record history
            4. Return new state
        """
        # 1. Allocation
        allocation = self.policy.allocate(self.state)

        # 2. Dynamics
        new_state = dynamics_step(
            state=self.state,
            allocation=allocation,
            external_signals=self._pending_signals or None,
            **self.dynamics_params,
        )
        new_state.timestamp = time.time()

        # 3. Bookkeeping
        self._history.append(self.state)
        self.state = new_state
        self._pending_signals = {}

        # 4. Notify adaptive policies
        if hasattr(self.policy, "update"):
            self.policy.update(self.state)

        return self.state

    def run(
        self,
        steps: int,
        signal_schedule: Optional[Dict[int, Dict[str, float]]] = None,
        callback: Optional[Callable[[SystemState], None]] = None,
    ) -> List[SystemState]:
        """
        Run for a fixed number of steps.

        Parameters
        ----------
        steps           : total steps to simulate
        signal_schedule : step_number → signals to inject at that step
        callback        : called after each step (logging, UI update, etc.)
        """
        schedule = signal_schedule or {}
        history  = []

        for _ in range(steps):
            current_step = self.state.step
            if current_step in schedule:
                self.ingest(schedule[current_step])

            state = self.step()
            history.append(state)

            if callback:
                callback(state)

        return history

    # ── State access ───────────────────────────────────────────

    @property
    def entropy(self) -> float:
        """Current H(t)."""
        return system_entropy(self.state)

    @property
    def history(self) -> List[SystemState]:
        return list(self._history)

    def entropy_history(self) -> List[float]:
        return [system_entropy(s) for s in self._history + [self.state]]

    def priority_allocations(self) -> List[Tuple[str, float]]:
        """Current allocations, sorted highest first."""
        return sorted(
            [(sid, s.allocation) for sid, s in self.state.sectors.items()],
            key=lambda x: -x[1],
        )

    def status_report(self) -> Dict[str, Any]:
        """
        Structured status for cognitive hub or UI.
        Delegates entirely to objective.status_report().
        """
        return make_status_report(
            state=self.state,
            history=self._history,
        )

    # ── Cognitive hub interface ────────────────────────────────

    def analyze(self, context: Optional[Dict] = None):
        """
        Request a cognitive report from the hub (if configured).
        Returns None if no hub is attached.
        """
        if self.cognitive_hub is None:
            return None
        return self.cognitive_hub.analyze(
            self.status_report(),
            context=context,
        )

    # ── Construction helpers ───────────────────────────────────

    @classmethod
    def from_dict(
        cls,
        sectors: Dict[str, Dict],
        coupling: Dict,
        policy: Optional[AllocationPolicy] = None,
        cognitive_hub=None,
        **dynamics_params,
    ) -> "ASASEngine":
        """
        Convenience constructor from plain dicts.

        Example
        -------
        engine = ASASEngine.from_dict(
            sectors={
                "Hbf":   {"risk": 0.9, "confidence": 0.7},
                "Messe": {"risk": 0.4, "confidence": 0.05},
            },
            coupling={("Messe", "Hbf"): 0.7},
            policy=SoftmaxPolicy(gamma=0.5),
        )
        """
        state = make_state(sectors=sectors, coupling=coupling)
        return cls(
            initial_state=state,
            policy=policy,
            cognitive_hub=cognitive_hub,
            **dynamics_params,
        )
