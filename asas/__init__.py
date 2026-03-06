# asas/__init__.py
"""
ASAS — Active Situational Awareness System

A cognitive situational awareness framework for cities.
Transforms fragmented signals into decision-grade intelligence.

Core imports:
    from asas import ASASEngine, make_state
    from asas.core.policy import SoftmaxPolicy, EqualPolicy
    from asas.cognitive.claude import ClaudeHub
"""
from .core import ASASEngine, make_state, SystemState, SoftmaxPolicy

__version__ = "0.1.0"
