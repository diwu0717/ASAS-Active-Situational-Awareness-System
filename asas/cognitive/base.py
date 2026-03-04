"""
asas/cognitive/base.py
======================
CognitiveHub: minimal abstract interface for LLM adapters.

This module defines the contract between ASAS core and any LLM backend.
It contains no prompt logic, no API calls, no model-specific code.

Implementing a backend
----------------------
    class GeminiHub(CognitiveHub):
        def analyze(self, status, context=None):
            prompt  = self._build_prompt(status, context)
            raw     = gemini_client.generate(prompt)
            return self._parse(raw)

See cognitive/gemini.py and cognitive/claude.py for full implementations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class CognitiveReport:
    """
    Structured output of one cognitive analysis cycle.

    Fields correspond to UI panels in the ASAS interface:
        operational_report   → current situation in plain language
        strategic_logic      → why allocation is as it is
        decision_deferrals   → items requiring human judgment
        actionable_gaps      → information needed to reduce uncertainty
        uncertainty_hotspots → highest residual uncertainty sectors
    """
    operational_report:   str       = ""
    strategic_logic:      str       = ""
    decision_deferrals:   List[str] = field(default_factory=list)
    actionable_gaps:      List[str] = field(default_factory=list)
    uncertainty_hotspots: List[str] = field(default_factory=list)
    confidence:           float     = 0.0
    raw_response:         str       = ""   # full LLM output for auditability

    def to_dict(self) -> Dict:
        return {
            "operational_report":   self.operational_report,
            "strategic_logic":      self.strategic_logic,
            "decision_deferrals":   self.decision_deferrals,
            "actionable_gaps":      self.actionable_gaps,
            "uncertainty_hotspots": self.uncertainty_hotspots,
            "confidence":           self.confidence,
        }


class CognitiveHub(ABC):
    """
    Abstract base for all ASAS cognitive backends.

    One method to implement: analyze().
    """

    @abstractmethod
    def analyze(
        self,
        status: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> CognitiveReport:
        """
        Analyze current ASAS state and produce intelligence report.

        Parameters
        ----------
        status  : from ASASEngine.status_report() — sector states,
                  allocations, entropy, hotspots, trend
        context : optional perception layer output
                  e.g. {"events": ["Ver.di strike active"],
                         "sensors": {"thermal": "elevated at Hbf"}}

        Returns
        -------
        CognitiveReport
        """
