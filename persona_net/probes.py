from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
import numpy as np

from .encoding import ExperienceEncoder, SCALAR_KEYS
from .network import PlasticRecurrentPersonaNet


def load_probes(path: str | Path) -> List[Dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["probes"]


def run_probes(
    net: PlasticRecurrentPersonaNet,
    encoder: ExperienceEncoder,
    probes: List[Dict],
    settle_ticks: int = 40,
    probe_ticks: int = 60,
) -> List[Dict]:
    results = []
    zero = np.zeros(encoder.input_dim, dtype=np.float32)

    for probe in probes:
        net.reset_fast_state(noise=0.01)
        for _ in range(settle_ticks):
            net.step(zero, reward=0.0, learn=False)

        scalars = {k: float(probe.get("scalars", {}).get(k, 0.0)) for k in SCALAR_KEYS}
        x = encoder.encode(probe["scenario"], scalars=scalars, action=None, outcome=0.0).vector
        accum = {a: 0.0 for a in net.action_populations}

        for _ in range(probe_ticks):
            net.step(x, reward=0.0, learn=False)
            scores = net.action_scores()
            for action, value in scores.items():
                accum[action] += value

        avg = {a: v / probe_ticks for a, v in accum.items()}
        ranked = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)
        results.append(
            {
                "id": probe["id"],
                "scenario": probe["scenario"],
                "action_probabilities": avg,
                "top_actions": ranked[:3],
                "expected_tendencies": probe.get("expected_tendencies", []),
                "source_role": probe.get("source_role", "holdout"),
            }
        )
    return results
