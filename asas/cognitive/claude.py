"""
asas/cognitive/claude.py
========================
Claude (Anthropic) adapter for ASAS CognitiveHub.

Reference implementation. Shows the pattern for any LLM adapter:
    1. Build prompt from status + context
    2. Call LLM API
    3. Parse structured response into CognitiveReport

Install: pip install anthropic
"""

from __future__ import annotations
import json
import os
from typing import Dict, Optional, Any

from .base import CognitiveHub, CognitiveReport

SYSTEM_PROMPT = """You are the cognitive reasoning layer of ASAS \
(Active Situational Awareness System).

Your role: transform quantitative sector state data into decision-grade \
situational intelligence for human operators.

You do NOT make decisions. You make the situation legible.

Rules:
- Be concise. Operators are under time pressure.
- Ground every claim in the provided data.
- Output valid JSON matching the schema below. No markdown, no preamble.

Required JSON schema:
{
  "operational_report":   "Current situation in 2-4 sentences.",
  "strategic_logic":      "Why allocation is as it is, 2-3 sentences.",
  "decision_deferrals":   ["Item requiring human judgment", ...],
  "actionable_gaps":      ["Specific info needed to reduce uncertainty", ...],
  "uncertainty_hotspots": ["Sector (reason)", ...],
  "confidence":           0.0
}"""


def _build_prompt(status: Dict, context: Optional[Dict]) -> str:
    lines = [
        f"Step: {status.get('step', '?')}",
        f"Global Entropy: {status.get('global_entropy')} "
        f"(trend: {status.get('entropy_trend', '?')})",
        "",
        "Sector States:",
    ]
    for sid, s in status.get("sector_states", {}).items():
        role = "SOURCE" if s.get("is_source") else ("SINK" if s.get("is_sink") else "")
        lines.append(
            f"  {sid} [{role}]: r={s['risk']} c={s['confidence']} "
            f"a={s['allocation']} score={s['base_score']}"
        )
    lines += ["", "Priority Allocations:"]
    for item in status.get("priority_allocations", []):
        lines.append(f"  {item['sector']}: {item['allocation_pct']}%")

    if context:
        lines += ["", "External Context:"]
        for k, v in context.items():
            if isinstance(v, list):
                lines.append(f"  {k}:")
                for item in v:
                    lines.append(f"    - {item}")
            else:
                lines.append(f"  {k}: {v}")

    return "\n".join(lines)


class ClaudeHub(CognitiveHub):
    """
    Anthropic Claude adapter.

    Parameters
    ----------
    api_key : Anthropic API key (default: ANTHROPIC_API_KEY env var)
    model   : Claude model string (default: claude-sonnet-4-6)
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,
        model:    str = "claude-sonnet-4-6",
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )
        self.model = model

    def analyze(
        self,
        status:  Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> CognitiveReport:
        prompt = _build_prompt(status, context)

        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text
        return self._parse(raw)

    def _parse(self, raw: str) -> CognitiveReport:
        text = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            d = json.loads(text)
            return CognitiveReport(
                operational_report=   d.get("operational_report", ""),
                strategic_logic=      d.get("strategic_logic", ""),
                decision_deferrals=   d.get("decision_deferrals", []),
                actionable_gaps=      d.get("actionable_gaps", []),
                uncertainty_hotspots= d.get("uncertainty_hotspots", []),
                confidence=           float(d.get("confidence", 0.0)),
                raw_response=raw,
            )
        except json.JSONDecodeError:
            return CognitiveReport(operational_report=raw, raw_response=raw)
