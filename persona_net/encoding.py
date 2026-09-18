from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import numpy as np

ACTIONS = [
    "explore",
    "challenge",
    "approach",
    "avoid",
    "cooperate",
    "dominate",
    "create",
    "persist",
    "conceal",
    "comply",
]

# These describe the environment/event, not Pretorius's personality.
SCALAR_KEYS = [
    "valence",
    "arousal",
    "social",
    "authority",
    "autonomy",
    "novelty",
    "achievement",
    "isolation",
    "threat",
    "intimacy",
    "control",
    "creation",
]


@dataclass
class EncodedExperience:
    vector: np.ndarray
    outcome: float


class ExperienceEncoder:
    """Deterministic, model-free encoder for developmental experiences.

    Text is projected with signed feature hashing. Generic environmental
    descriptors and an optional efference-copy action channel are appended.
    No Pretorius-specific trait dimensions are present.
    """

    def __init__(self, sensory_dim: int = 512):
        self.sensory_dim = int(sensory_dim)
        self.scalar_offset = self.sensory_dim
        self.action_offset = self.scalar_offset + len(SCALAR_KEYS)
        self.input_dim = self.action_offset + len(ACTIONS)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        toks = re.findall(r"[a-z0-9']+", text.lower())
        return toks + [f"{a}::{b}" for a, b in zip(toks, toks[1:])]

    def _hash_index_sign(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        idx = value % self.sensory_dim
        sign = 1.0 if ((value >> 8) & 1) else -1.0
        return idx, sign

    def encode(
        self,
        text: str,
        scalars: Optional[Dict[str, float]] = None,
        action: Optional[str] = None,
        outcome: float = 0.0,
    ) -> EncodedExperience:
        x = np.zeros(self.input_dim, dtype=np.float32)
        tokens = self._tokens(text)
        for tok in tokens:
            idx, sign = self._hash_index_sign(tok)
            x[idx] += sign

        if tokens:
            norm = float(np.linalg.norm(x[: self.sensory_dim]))
            if norm > 1e-8:
                x[: self.sensory_dim] /= norm

        scalars = scalars or {}
        for i, key in enumerate(SCALAR_KEYS):
            x[self.scalar_offset + i] = float(np.clip(scalars.get(key, 0.0), -1.0, 1.0))

        if action is not None:
            if action not in ACTIONS:
                raise ValueError(f"Unknown action {action!r}. Valid actions: {ACTIONS}")
            x[self.action_offset + ACTIONS.index(action)] = 1.0

        return EncodedExperience(vector=x, outcome=float(np.clip(outcome, -1.0, 1.0)))
