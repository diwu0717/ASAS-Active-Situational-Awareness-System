"""
examples/frankfurt_strike/scenario.py
======================================
Frankfurt Transport Strike — Reference Scenario

Reproduces the situation from the ASAS demo (Feb 2, 2026):
Ver.di nationwide municipal transport strike paralyzing Frankfurt's
bus, tram and U-Bahn. S-Bahn/Regional trains operational but overcrowded.

This scenario demonstrates the core ASAS value proposition:
risk arises from the interaction of external events (strike + construction
+ weather) across sectors, not from any single anomaly. No individual
sensor sees the full picture. ASAS integrates them.

Run:
    python examples/frankfurt_strike/scenario.py
    python examples/frankfurt_strike/scenario.py --llm claude
    python examples/frankfurt_strike/scenario.py --llm gemini
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from asas.core.engine import ASASEngine
from asas.core.policy import SoftmaxPolicy
from asas.core.objective import system_entropy


# ── Sector definitions ─────────────────────────────────────────────────────────

SECTORS = {
    "Hbf":     {"risk": 0.55, "confidence": 0.70},   # Frankfurt Hauptbahnhof
    "Bridges": {"risk": 0.40, "confidence": 0.50},   # Main River Crossings
    "Messe":   {"risk": 0.25, "confidence": 0.30},   # Messe Frankfurt
    "Airport": {"risk": 0.35, "confidence": 0.45},   # Frankfurt Airport FRA
    "Zeil":    {"risk": 0.30, "confidence": 0.40},   # Zeil District
}

# Network topology: how disruption propagates between sectors
COUPLING = {
    ("Hbf",     "Bridges"):  0.50,  # Hbf overflow → bridge traffic
    ("Hbf",     "Zeil"):     0.30,  # Stranded passengers → Zeil crowds
    ("Bridges", "Airport"):  0.60,  # Road congestion → airport access
    ("Messe",   "Hbf"):      0.40,  # Messe events → Hbf pressure
    ("Messe",   "Airport"):  0.30,  # Messe logistics → airport freight
}

# ── Event schedule ─────────────────────────────────────────────────────────────

# Step 3: Strike confirmed — all municipal transport suspended
STRIKE_ALERT = {
    "Hbf":     +0.40,   # All commuters forced onto S-Bahn/Regional rail
    "Bridges": +0.20,   # Modal shift to private cars
    "Airport": +0.15,   # S8/S9 disruption (Mainz construction compound)
    "Zeil":    +0.10,   # Stranded crowds
    "Messe":   +0.05,   # Logistics disruption
}

# Step 8: Thermal sensors report elevated platform density at Hbf
THERMAL_ALERT = {"Hbf": +0.15}

SIGNAL_SCHEDULE = {3: STRIKE_ALERT, 8: THERMAL_ALERT}

# ── Context for cognitive hub ──────────────────────────────────────────────────

CONTEXT = {
    "events": [
        "Ver.di strike confirmed: VGF bus/tram/U-Bahn fully suspended",
        "S-Bahn (DB) operational but reporting extreme overcrowding",
        "Railway construction at Mainz impacting S8/S9 Airport connections",
        "Messe Frankfurt: setup underway for Ambiente/CreativeWorld (Feb 6)",
    ],
    "weather": "Overcast, rising pollution from traffic congestion",
    "sensors": {
        "thermal":  "Elevated signatures at Hbf platforms (Delta 83%)",
        "rf":       "Significant congestion shift at Hbf (Delta 43%)",
        "flow":     "Elevated density at Main River Crossings (Delta 51%)",
    },
}


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    cognitive_hub=None,
    steps: int = 20,
    gamma: float = 0.5,
    outflow_weight: float = 0.5,
    natural_decay: float = 0.04,
):
    engine = ASASEngine.from_dict(
        sectors=SECTORS,
        coupling=COUPLING,
        policy=SoftmaxPolicy(gamma=gamma, outflow_weight=outflow_weight),
        cognitive_hub=cognitive_hub,
        natural_decay=natural_decay,
        mitigation_strength=0.25,
        learning_rate=0.15,
        forgetting_rate=0.03,
    )

    print("=" * 60)
    print("ASAS — Frankfurt Transport Strike Scenario")
    print("=" * 60)

    for i in range(steps):
        step_num = engine.state.step
        if step_num in SIGNAL_SCHEDULE:
            engine.ingest(SIGNAL_SCHEDULE[step_num])
            print(f"\n  ⚡ Step {step_num}: event injected "
                  f"({list(SIGNAL_SCHEDULE[step_num].keys())})")

        state = engine.step()
        allocs = "  ".join(
            f"{sid}={s.allocation*100:.0f}%"
            for sid, s in state.sectors.items()
        )
        print(f"Step {state.step:02d} | H={system_entropy(state):.3f} | {allocs}")

        # Cognitive analysis at step 10
        if i == 9 and cognitive_hub:
            print("\n" + "─" * 60)
            print("COGNITIVE HUB ANALYSIS")
            print("─" * 60)
            report = engine.analyze(context=CONTEXT)
            print(f"\nOPERATIONAL REPORT:\n{report.operational_report}")
            print(f"\nSTRATEGIC LOGIC:\n{report.strategic_logic}")
            if report.decision_deferrals:
                print("\nDECISION DEFERRALS:")
                for d in report.decision_deferrals:
                    print(f"  • {d}")
            if report.actionable_gaps:
                print("\nACTIONABLE GAPS:")
                for g in report.actionable_gaps:
                    print(f"  • {g}")
            print("─" * 60 + "\n")

    print(f"\nFinal H(t) = {engine.entropy:.4f}")
    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASAS Frankfurt Strike Scenario",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--llm",    choices=["claude", "gemini", "none"],
                        default="none", help="Cognitive hub backend")
    parser.add_argument("--steps",  type=int,   default=20,
                        help="Number of simulation steps")
    parser.add_argument("--gamma",  type=float, default=0.5,
                        help="SoftmaxPolicy temperature (γ→∞ = Equal, γ→0 = greedy)")
    parser.add_argument("--outflow-weight", type=float, default=0.5,
                        dest="outflow_weight",
                        help="Weight of outflow signal in allocation score")
    parser.add_argument("--decay",  type=float, default=0.04,
                        help="Natural risk decay rate per step")
    args = parser.parse_args()

    hub = None
    if args.llm == "claude":
        from asas.cognitive.claude import ClaudeHub
        hub = ClaudeHub()
    elif args.llm == "gemini":
        try:
            from asas.cognitive.gemini import GeminiHub
            hub = GeminiHub()
        except ImportError:
            print("pip install google-generativeai")
            sys.exit(1)

    run(
        cognitive_hub=hub,
        steps=args.steps,
        gamma=args.gamma,
        outflow_weight=args.outflow_weight,
        natural_decay=args.decay,
    )
