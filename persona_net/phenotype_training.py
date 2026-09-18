from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json
import numpy as np

from .encoding import ACTIONS, ExperienceEncoder, SCALAR_KEYS
from .network import PlasticRecurrentPersonaNet


@dataclass
class PhenotypeTrainingReport:
    # `ticks` is retained for v0.3 compatibility. In this curriculum each tick
    # is exactly one call to net.step, but v0.4 code should use neural_steps.
    ticks: int
    presentations: int
    neural_steps: int
    start_neural_step: int
    end_neural_step: int


class PhenotypeCurriculum:
    """Train the same generic neural architecture from mature behavioral targets.

    This curriculum contains no biography. Each training item specifies a novel
    situation and an external target distribution over generic actions.
    """

    def __init__(self, items: List[Dict], encoder: ExperienceEncoder):
        self.items = list(items)
        self.encoder = encoder

    @classmethod
    def from_json(cls, path: str | Path, encoder: ExperienceEncoder) -> 'PhenotypeCurriculum':
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(payload['training_items'], encoder)

    @staticmethod
    def _normalize_target(raw: Dict[str, float]) -> Dict[str, float]:
        target = {a: max(float(raw.get(a, 0.0)), 0.0) for a in ACTIONS}
        total = sum(target.values())
        if total <= 0:
            raise ValueError('Every phenotype training item needs positive target action mass.')
        return {a: v / total for a, v in target.items()}

    def run(self, net: PlasticRecurrentPersonaNet, total_ticks: int, progress_every: int = 10000) -> PhenotypeTrainingReport:
        start_neural_step = int(net.tick)
        if not self.items:
            return PhenotypeTrainingReport(0, 0, 0, start_neural_step, start_neural_step)

        done = 0
        presentations = 0
        cursor = 0
        zero_action = slice(self.encoder.action_offset, self.encoder.input_dim)

        while done < total_ticks:
            item = self.items[cursor % len(self.items)]
            cursor += 1
            presentations += 1
            scalars = {k: float(item.get('scalars', {}).get(k, 0.0)) for k in SCALAR_KEYS}
            target = self._normalize_target(item['target_actions'])

            context = self.encoder.encode(item['scenario'], scalars=scalars, action=None, outcome=0.0).vector
            x_context = context.copy()
            x_context[zero_action] = 0.0

            net.step(x_context, reward=0.0, learn=False)
            net.learn_motor_distribution(target)
            done += 1
            if done >= total_ticks:
                break

            teaching = context.copy()
            for ai, action in enumerate(ACTIONS):
                teaching[self.encoder.action_offset + ai] = target[action]
            net.step(teaching, reward=1.0, learn=True)
            done += 1

            if progress_every and done % progress_every < 2:
                print(f'phenotype synthesis: {done:,}/{total_ticks:,} neural steps')

        end_neural_step = int(net.tick)
        return PhenotypeTrainingReport(
            ticks=done,
            presentations=presentations,
            neural_steps=end_neural_step - start_neural_step,
            start_neural_step=start_neural_step,
            end_neural_step=end_neural_step,
        )
