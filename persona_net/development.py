from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
import json
import numpy as np

from .encoding import ExperienceEncoder, SCALAR_KEYS
from .network import PlasticRecurrentPersonaNet


@dataclass
class DevelopmentReport:
    ticks: int
    event_ticks: Dict[str, int]


class BiographyCurriculum:
    """Chronological developmental replay.

    `exposure_weight` controls how much simulated developmental time an event
    or phase receives. Long periods can therefore occupy more ticks without
    inventing undocumented biographical episodes.
    """

    def __init__(self, events: List[Dict], encoder: ExperienceEncoder):
        self.events = sorted(events, key=lambda e: (float(e["year"]), e["id"]))
        self.encoder = encoder

    @classmethod
    def from_json(cls, path: str | Path, encoder: ExperienceEncoder) -> "BiographyCurriculum":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["events"], encoder)

    def allocations(self, total_ticks: int) -> Dict[str, int]:
        weights = np.asarray([max(float(e.get("exposure_weight", 1.0)), 0.01) for e in self.events])
        raw = weights / weights.sum() * int(total_ticks)
        alloc = np.maximum(1, np.floor(raw).astype(int))
        remaining = int(total_ticks) - int(alloc.sum())
        order = np.argsort(-(raw - np.floor(raw)))
        for idx in order[: max(remaining, 0)]:
            alloc[idx] += 1
        if remaining < 0:
            for idx in order[::-1]:
                if remaining == 0:
                    break
                if alloc[idx] > 1:
                    alloc[idx] -= 1
                    remaining += 1
        return {e["id"]: int(a) for e, a in zip(self.events, alloc)}

    def run(
        self,
        net: PlasticRecurrentPersonaNet,
        total_ticks: int,
        input_noise: float = 0.03,
        consolidation_fraction: float = 0.08,
        progress_every: int = 10000,
    ) -> DevelopmentReport:
        allocations = self.allocations(total_ticks)
        done = 0

        for event in self.events:
            ticks = allocations[event["id"]]
            scalars = {k: float(event.get("scalars", {}).get(k, 0.0)) for k in SCALAR_KEYS}
            encoded = self.encoder.encode(
                event["narrative"],
                scalars=scalars,
                action=event.get("action"),
                outcome=float(event.get("outcome", 0.0)),
            )
            base_x = encoded.vector
            reward = encoded.outcome

            active_ticks = max(1, int(round(ticks * (1.0 - consolidation_fraction))))
            consolidation_ticks = max(0, ticks - active_ticks)

            for _ in range(active_ticks):
                x_full = base_x
                if input_noise > 0:
                    x_full = base_x + net.rng.normal(
                        0.0, input_noise, base_x.shape
                    ).astype(np.float32)

                # First expose the network to context without being told the
                # action. The motor decoder learns from this distributed state.
                x_context = x_full.copy()
                x_context[self.encoder.action_offset :] = 0.0
                net.step(x_context, reward=0.0, learn=False)
                if event.get("action"):
                    net.learn_motor(event["action"])

                # Then supply the actual action as efference copy and allow
                # recurrent synapses to associate context, action, and outcome.
                net.step(x_full, reward=reward, learn=True)

                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"development: {done:,}/{total_ticks:,} ticks")

            # Offline-style replay: preserve the experience but remove the
            # explicit action teaching channel and external reward.
            if consolidation_ticks:
                replay = base_x.copy()
                replay[self.encoder.action_offset :] = 0.0
                for _ in range(consolidation_ticks):
                    x = replay
                    if input_noise > 0:
                        x = replay + net.rng.normal(
                            0.0, input_noise * 0.5, replay.shape
                        ).astype(np.float32)
                    net.step(x, reward=0.0, learn=True)
                    done += 1
                    if progress_every and done % progress_every == 0:
                        print(f"development: {done:,}/{total_ticks:,} ticks")

        return DevelopmentReport(ticks=done, event_ticks=allocations)
