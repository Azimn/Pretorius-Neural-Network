"""BioCircuit BC00: executable sparse associative circuit proof of concept.

This is a *small functional kernel*, not a fly-brain model or a persona.
It isolates local inhibitory competition using exactly matched sensory
projection wiring, plastic output synapse count, data and training steps.
Existing persona_net and Doctor Lives implementations remain unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy import sparse

CHECKPOINT_SCHEMA = 1


@dataclass(frozen=True)
class CircuitConfig:
    neurons: int = 4096
    sensory_dim: int = 256
    fan_in: int = 12
    active: int = 128
    compartments: int = 4
    actions: int = 4
    learning_rate: float = 0.7
    seed: int = 31

    def validate(self) -> None:
        if (self.neurons < 128 or self.compartments < 2
                or self.neurons % self.compartments
                or self.active < self.compartments
                or self.active % self.compartments
                or self.active > self.neurons
                or self.sensory_dim < 64
                or self.sensory_dim % self.compartments
                or not 1 <= self.fan_in <= self.sensory_dim // self.compartments
                or self.actions < 2 or not 0 < self.learning_rate <= 2):
            raise ValueError("incompatible BioCircuit configuration")


class Circuit:
    """Sparse projection, inhibitory competition and plastic action synapses.

    'local' selects k neurons inside each compartment; 'global' selects k
    from the whole population. Every other trainable and fixed parameter is
    identical across the paired conditions given the same configuration.

    Weights are the ONLY learned event-to-action state. There is no trainable
    motor decoder, episodic lookup or external oracle codebook.
    """

    def __init__(self, config: CircuitConfig, competition: str):
        config.validate()
        if competition not in ("local", "global"):
            raise ValueError("competition must be 'local' or 'global'")
        self.config = config
        self.competition = competition
        rng = np.random.default_rng(config.seed)
        width = config.neurons // config.compartments
        channels = config.sensory_dim // config.compartments
        # Both conditions share the exact same compartment-biased sensory
        # projection; ONLY the inhibitory competition policy changes.
        pre = np.empty((config.neurons, config.fan_in), dtype=np.int32)
        for compartment in range(config.compartments):
            start, end = compartment * width, (compartment + 1) * width
            for neuron in range(start, end):
                pre[neuron] = rng.choice(
                    np.arange(compartment * channels,
                              (compartment + 1) * channels),
                    size=config.fan_in, replace=False
                )
        rows = np.repeat(np.arange(config.neurons), config.fan_in)
        cols = pre.ravel()
        data = np.ones(rows.shape[0], dtype=np.float32) / np.sqrt(config.fan_in)
        self.projection = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(config.neurons, config.sensory_dim),
            dtype=np.float32
        )
        self.weights = np.zeros((config.actions, config.neurons), dtype=np.float32)
        self.presentations = 0
        self.plastic_updates = 0

    def _activity(self, stimulus: np.ndarray) -> np.ndarray:
        x = np.asarray(stimulus, dtype=np.float32)
        if x.shape != (self.config.sensory_dim,):
            raise ValueError("stimulus has incorrect shape")
        if not np.isfinite(x).all():
            raise ValueError("non-finite stimulus")
        potentials = np.asarray(self.projection @ x, dtype=np.float32)
        chosen: list[np.ndarray] = []
        if self.competition == "local":
            width = self.config.neurons // self.config.compartments
            take = self.config.active // self.config.compartments
            for i in range(self.config.compartments):
                start = i * width
                segment = potentials[start:start + width]
                idx = np.argpartition(segment, -take)[-take:] + start
                chosen.append(idx)
        else:
            chosen.append(np.argpartition(potentials, -self.config.active)[
                -self.config.active:
            ])
        result = np.zeros(self.config.neurons, dtype=np.float32)
        result[np.concatenate(chosen)] = 1.0 / np.sqrt(self.config.active)
        return result

    def decision_scores(self, stimulus: np.ndarray) -> np.ndarray:
        return self.weights @ self._activity(stimulus)

    def predict(self, stimulus: np.ndarray) -> int:
        return int(np.argmax(self.decision_scores(stimulus)))

    def learn(self, stimulus: np.ndarray, correct_action: int) -> None:
        if not 0 <= correct_action < self.config.actions:
            raise ValueError("invalid action")
        firing = self._activity(stimulus)
        active = np.flatnonzero(firing)
        rate = self.config.learning_rate
        # A locally specified supervised reward gate: strengthen rewarded
        # MBON-like pathway and weaken its competing outputs. It uses the
        # action label ONLY at learning time, never during inference.
        self.weights[:, active] -= (
            rate / (self.config.actions - 1)
        ) * firing[active][None, :]
        self.weights[correct_action, active] += (
            rate + rate / (self.config.actions - 1)
        ) * firing[active]
        self.presentations += 1
        self.plastic_updates += len(active) * self.config.actions

    def lesion(self, fraction: float = 1.0) -> None:
        """Destroy a specified fraction of learned output synaptic weights."""
        if not 0 <= fraction <= 1:
            raise ValueError("invalid lesion fraction")
        if fraction == 1:
            self.weights.fill(0)
            return
        # Reproducible structural pattern independent of data labels.
        rng = np.random.default_rng(self.config.seed + 7703)
        kill = rng.choice(
            self.config.neurons,
            size=int(round(self.config.neurons * fraction)),
            replace=False
        )
        self.weights[:, kill] = 0

    def save(self, path: str | Path) -> None:
        metadata = json.dumps({
            "schema": CHECKPOINT_SCHEMA,
            "config": asdict(self.config),
            "competition": self.competition,
            "presentations": self.presentations,
            "plastic_updates": self.plastic_updates,
        }, sort_keys=True)
        np.savez_compressed(path, metadata=metadata, weights=self.weights)

    @classmethod
    def load(cls, path: str | Path) -> "Circuit":
        with np.load(path, allow_pickle=False) as raw:
            metadata = json.loads(raw["metadata"].item())
            if metadata["schema"] != CHECKPOINT_SCHEMA:
                raise ValueError("unsupported checkpoint")
            cfg = CircuitConfig(**metadata["config"])
            circuit = cls(cfg, metadata["competition"])
            weights = raw["weights"]
            if weights.shape != circuit.weights.shape or not np.isfinite(weights).all():
                raise ValueError("corrupt synaptic checkpoint")
            circuit.weights[:] = weights
            circuit.presentations = int(metadata["presentations"])
            circuit.plastic_updates = int(metadata["plastic_updates"])
            return circuit


@dataclass(frozen=True)
class Experience:
    stimulus: np.ndarray
    noisy_query: np.ndarray
    action: int


def curriculum(seed: int, config: CircuitConfig, items: int = 128) -> list[Experience]:
    """Fiction-free synthetic cause/outcome associations, no leaked labels.

    Four context categories use disjoint blocks of sensor channels. Each
    example samples 18 within-context plus 6 background signals. Queries
    hide approximately one third of observed features and add distractors.
    """
    config.validate()
    if items <= 0:
        raise ValueError("items must be positive")
    rng = np.random.default_rng(seed + 1009)
    block = config.sensory_dim // config.compartments
    examples = []
    for _ in range(items):
        context = int(rng.integers(config.compartments))
        local = np.arange(context * block, (context + 1) * block)
        foreground = rng.choice(local, size=min(18, block), replace=False)
        bg_pool = np.setdiff1d(np.arange(config.sensory_dim), local)
        background = rng.choice(bg_pool, size=min(6, len(bg_pool)), replace=False)
        active = np.concatenate((foreground, background))
        query_active = rng.choice(active, size=max(2, len(active) * 2 // 3), replace=False)
        distractors = rng.choice(
            np.setdiff1d(np.arange(config.sensory_dim), active),
            size=max(1, len(active) // 5), replace=False
        )
        x = np.zeros(config.sensory_dim, dtype=np.float32)
        q = np.zeros(config.sensory_dim, dtype=np.float32)
        x[active] = 1 / np.sqrt(len(active))
        indices = np.concatenate((query_active, distractors))
        q[indices] = 1 / np.sqrt(len(indices))
        # Outcome is randomized independently of sensor features.
        action = int(rng.integers(config.actions))
        examples.append(Experience(x, q, action))
    return examples


def accuracy(network: Circuit, samples: list[Experience], noisy: bool) -> float:
    if not samples:
        raise ValueError("empty evaluation")
    return float(np.mean([
        network.predict(example.noisy_query if noisy else example.stimulus)
        == example.action for example in samples
    ]))


def assay(seed: int, neurons: int = 4096, items: int = 128,
          epochs: int = 4, active: int | None = None) -> dict:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    cfg = CircuitConfig(
        neurons=neurons, seed=seed,
        active=active if active is not None else max(16, neurons // 32)
    )
    samples = curriculum(seed, cfg, items)
    result = {
        "seed": seed,
        "neurons": neurons,
        "items": items,
        "epochs": epochs,
        "task": "random synthetic sensor-context to action associations",
        "caveat": (
            "No Pretorius autobiography, semantic interpretation, FlyWire "
            "topology, recurrent state, or biological dopamine physiology. "
            "A 4-class fixed neural action population is learned directly."
        ),
        "conditions": {},
    }
    models = {mode: Circuit(cfg, mode) for mode in ("local", "global")}
    for mode, model in models.items():
        before = accuracy(model, samples, noisy=True)
        for _ in range(epochs):
            for sample in samples:
                model.learn(sample.stimulus, sample.action)
        intact = accuracy(model, samples, noisy=False)
        noisy = accuracy(model, samples, noisy=True)
        checkpoint = model.weights.copy()
        model.lesion()
        lesioned = accuracy(model, samples, noisy=True)
        model.weights[:] = checkpoint
        replay_equal = all(
            model.predict(s.noisy_query) == int(np.argmax(checkpoint @ model._activity(s.noisy_query)))
            for s in samples
        )
        result["conditions"][mode] = {
            "before_accuracy": round(before, 6),
            "trained_accuracy": round(intact, 6),
            "partial_cue_accuracy": round(noisy, 6),
            "lesioned_partial_cue_accuracy": round(lesioned, 6),
            "recovered_checkpoint_exact": bool(replay_equal),
            "active_per_input": cfg.active,
            "fixed_projection_edges": int(model.projection.nnz),
            "trainable_action_synapses": int(model.weights.size),
            "presentations": model.presentations,
            "plastic_updates": model.plastic_updates,
        }
    assert np.array_equal(
        models["local"].projection.indices, models["global"].projection.indices
    )
    assert np.array_equal(
        models["local"].projection.data, models["global"].projection.data
    )
    result["matched_projection_exact"] = True
    return result
