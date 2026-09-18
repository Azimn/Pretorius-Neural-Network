from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable
import json
import numpy as np
from scipy import sparse

from .encoding import ACTIONS, ExperienceEncoder


@dataclass
class StepStats:
    mean_rate: float
    max_rate: float


class PlasticRecurrentPersonaNet:
    """Sparse recurrent rate network with generic developmental plasticity.

    Design goals:
      * thousands of neurons on commodity CPU hardware
      * Dale-like excitatory/inhibitory sign constraint
      * recurrent state and sparse connectivity
      * local Hebbian plasticity
      * reward-modulated eligibility traces
      * slow homeostatic bias adaptation
      * generic action populations used only as a behavioral readout

    The network contains no named personality traits.
    """

    def __init__(self, cfg: Dict, encoder: ExperienceEncoder):
        self.cfg = dict(cfg)
        self.encoder = encoder
        self.n = int(cfg["neurons"])
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))

        self.target_rate = float(cfg["target_rate"])
        self.tau = float(cfg["tau"])
        self.dt = float(cfg["dt"])
        self.global_inhibition = float(cfg["global_inhibition"])
        self.plasticity_interval = int(cfg["plasticity_interval"])
        self.hebb_lr = float(cfg["hebb_lr"])
        self.reward_lr = float(cfg["reward_lr"])
        self.weight_decay = float(cfg["weight_decay"])
        self.homeostatic_lr = float(cfg["homeostatic_lr"])
        self.eligibility_decay = float(cfg["eligibility_decay"])
        self.max_abs_weight = float(cfg["max_abs_weight"])

        self.excitatory = self.rng.random(self.n) < float(cfg["excitatory_fraction"])

        self.W = self._make_recurrent()
        self.post_idx = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.W.indptr))
        self.pre_idx = self.W.indices.astype(np.int32, copy=False)
        self.eligibility = np.zeros_like(self.W.data, dtype=np.float32)

        self.action_populations = self._make_action_populations()
        self.Win = self._make_input_matrix()

        # Generic learned motor decoder. It maps distributed network state to
        # action tendencies. It is not persona-specific and is trained only
        # from actions actually taken in the developmental biography.
        self.motor_w = self.rng.normal(
            0.0, 0.01, size=(len(ACTIONS), self.n)
        ).astype(np.float32)
        self.motor_b = np.zeros(len(ACTIONS), dtype=np.float32)
        self.motor_lr = float(cfg.get("motor_lr", 0.02))
        self.motor_weight_decay = float(cfg.get("motor_weight_decay", 0.00002))
        self.motor_temperature = float(cfg.get("motor_temperature", 0.30))

        self.v = np.zeros(self.n, dtype=np.float32)
        self.rate = np.full(self.n, self.target_rate, dtype=np.float32)
        self.bias = np.full(self.n, -2.45, dtype=np.float32)
        self.tick = 0

    def _make_recurrent(self) -> sparse.csr_matrix:
        k = int(self.cfg["avg_recurrent_degree"])
        m = self.n * k
        post = np.repeat(np.arange(self.n, dtype=np.int32), k)
        pre = self.rng.integers(0, self.n, size=m, dtype=np.int32)

        self_mask = pre == post
        while np.any(self_mask):
            pre[self_mask] = self.rng.integers(0, self.n, size=int(self_mask.sum()), dtype=np.int32)
            self_mask = pre == post

        scale = float(self.cfg["recurrent_scale"]) / np.sqrt(max(k, 1))
        mag = self.rng.lognormal(mean=-1.0, sigma=0.45, size=m).astype(np.float32) * scale
        sign = np.where(self.excitatory[pre], 1.0, -1.0).astype(np.float32)
        data = mag * sign

        W = sparse.csr_matrix((data, (post, pre)), shape=(self.n, self.n), dtype=np.float32)
        W.sum_duplicates()
        return W

    def _make_action_populations(self) -> Dict[str, np.ndarray]:
        size = min(int(self.cfg["action_population_size"]), max(8, self.n // (len(ACTIONS) * 2)))
        available = np.arange(self.n, dtype=np.int32)
        self.rng.shuffle(available)
        pops = {}
        cursor = 0
        for action in ACTIONS:
            if cursor + size > len(available):
                available = np.arange(self.n, dtype=np.int32)
                self.rng.shuffle(available)
                cursor = 0
            pops[action] = np.sort(available[cursor : cursor + size])
            cursor += size
        return pops

    def _make_input_matrix(self) -> sparse.csr_matrix:
        input_dim = self.encoder.input_dim
        k = int(self.cfg["input_degree"])
        m = self.n * k
        rows = np.repeat(np.arange(self.n, dtype=np.int32), k)
        cols = self.rng.integers(0, input_dim, size=m, dtype=np.int32)
        data = (
            self.rng.normal(
                0.0,
                float(self.cfg["input_scale"]) / np.sqrt(max(k, 1)),
                size=m,
            )
            .astype(np.float32)
        )

        extra_rows = []
        extra_cols = []
        extra_data = []
        teach = float(self.cfg["action_teaching_scale"])
        for ai, action in enumerate(ACTIONS):
            col = self.encoder.action_offset + ai
            pop = self.action_populations[action]
            extra_rows.extend(pop.tolist())
            extra_cols.extend([col] * len(pop))
            extra_data.extend([teach] * len(pop))

        rows = np.concatenate([rows, np.asarray(extra_rows, dtype=np.int32)])
        cols = np.concatenate([cols, np.asarray(extra_cols, dtype=np.int32)])
        data = np.concatenate([data, np.asarray(extra_data, dtype=np.float32)])

        Win = sparse.csr_matrix((data, (rows, cols)), shape=(self.n, input_dim), dtype=np.float32)
        Win.sum_duplicates()
        return Win

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        z = np.clip(x, -10.0, 10.0)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32, copy=False)

    def reset_fast_state(self, noise: float = 0.0) -> None:
        self.v.fill(0.0)
        self.rate.fill(self.target_rate)
        if noise > 0:
            self.v += self.rng.normal(0.0, noise, self.n).astype(np.float32)

    def step(self, x: np.ndarray, reward: float = 0.0, learn: bool = True) -> StepStats:
        syn = self.W.dot(self.rate)
        ext = self.Win.dot(x)

        excess = max(float(self.rate.mean()) - self.target_rate, 0.0)
        global_term = self.global_inhibition * excess

        self.v += (self.dt / self.tau) * (-self.v + syn + ext + self.bias - global_term)
        self.rate = self._sigmoid(self.v)
        self.tick += 1

        if learn and self.tick % self.plasticity_interval == 0:
            centered_pre = self.rate[self.pre_idx] - self.target_rate
            centered_post = self.rate[self.post_idx] - self.target_rate
            corr = centered_pre * centered_post

            self.eligibility *= self.eligibility_decay
            self.eligibility += corr.astype(np.float32)

            delta = (
                self.hebb_lr * corr
                + self.reward_lr * float(reward) * self.eligibility
                - self.weight_decay * self.W.data
            )
            self.W.data += delta.astype(np.float32)

            pre_exc = self.excitatory[self.pre_idx]
            self.W.data[pre_exc] = np.clip(
                self.W.data[pre_exc], 0.0, self.max_abs_weight
            )
            self.W.data[~pre_exc] = np.clip(
                self.W.data[~pre_exc], -self.max_abs_weight, 0.0
            )

            self.bias += self.homeostatic_lr * (self.target_rate - self.rate)

        return StepStats(float(self.rate.mean()), float(self.rate.max()))

    def _motor_features(self) -> np.ndarray:
        f = (self.rate - self.target_rate).astype(np.float32, copy=True)
        norm = float(np.linalg.norm(f))
        if norm > 1e-8:
            f /= norm
        return f

    def action_scores(self) -> Dict[str, float]:
        f = self._motor_features()
        logits = self.motor_w.dot(f) + self.motor_b
        logits = logits.astype(np.float64)
        logits -= logits.max()
        probs = np.exp(logits / max(self.motor_temperature, 1e-4))
        probs /= probs.sum()
        return {a: float(p) for a, p in zip(ACTIONS, probs)}

    def learn_motor(self, action: str) -> None:
        if action not in ACTIONS:
            return
        f = self._motor_features()
        logits = self.motor_w.dot(f) + self.motor_b
        logits = logits.astype(np.float64)
        logits -= logits.max()
        probs = np.exp(logits / max(self.motor_temperature, 1e-4))
        probs /= probs.sum()

        target = np.zeros(len(ACTIONS), dtype=np.float64)
        target[ACTIONS.index(action)] = 1.0
        error = (target - probs).astype(np.float32)

        self.motor_w *= (1.0 - self.motor_weight_decay)
        self.motor_w += self.motor_lr * np.outer(error, f).astype(np.float32)
        self.motor_b += (self.motor_lr * 0.1) * error


    def learn_motor_distribution(self, target_probs: Dict[str, float]) -> None:
        """Train the generic motor decoder toward a soft target distribution."""
        f = self._motor_features()
        logits = self.motor_w.dot(f) + self.motor_b
        logits = logits.astype(np.float64)
        logits -= logits.max()
        probs = np.exp(logits / max(self.motor_temperature, 1e-4))
        probs /= probs.sum()

        target = np.zeros(len(ACTIONS), dtype=np.float64)
        for action, value in target_probs.items():
            if action not in ACTIONS:
                raise ValueError(f"Unknown action {action!r}. Valid actions: {ACTIONS}")
            target[ACTIONS.index(action)] = max(float(value), 0.0)
        total = float(target.sum())
        if total <= 0.0:
            raise ValueError("Target action distribution must contain positive mass.")
        target /= total

        error = (target - probs).astype(np.float32)
        self.motor_w *= (1.0 - self.motor_weight_decay)
        self.motor_w += self.motor_lr * np.outer(error, f).astype(np.float32)
        self.motor_b += (self.motor_lr * 0.1) * error

    def recurrent_summary(self) -> Dict[str, float]:
        data = self.W.data
        return {
            "neurons": int(self.n),
            "recurrent_synapses": int(data.size),
            "mean_abs_weight": float(np.mean(np.abs(data))),
            "std_abs_weight": float(np.std(np.abs(data))),
            "mean_rate": float(self.rate.mean()),
            "weight_l1": float(np.sum(np.abs(data))),
        }


    @classmethod
    def load(cls, path: str | Path, cfg: Dict, encoder: ExperienceEncoder) -> "PlasticRecurrentPersonaNet":
        """Recreate deterministic fixed wiring, then restore learned mature state."""
        payload = np.load(Path(path), allow_pickle=False)
        if "cfg_json" in payload.files:
            saved_cfg = json.loads(str(payload["cfg_json"].item()))
            cfg = saved_cfg
        net = cls(cfg, encoder)
        net.v = payload["v"].astype(np.float32, copy=True)
        net.rate = payload["rate"].astype(np.float32, copy=True)
        net.bias = payload["bias"].astype(np.float32, copy=True)
        net.W = sparse.csr_matrix(
            (
                payload["w_data"].astype(np.float32, copy=True),
                payload["w_indices"].astype(np.int32, copy=True),
                payload["w_indptr"].astype(np.int32, copy=True),
            ),
            shape=(net.n, net.n),
            dtype=np.float32,
        )
        net.post_idx = np.repeat(np.arange(net.n, dtype=np.int32), np.diff(net.W.indptr))
        net.pre_idx = net.W.indices.astype(np.int32, copy=False)
        net.excitatory = payload["excitatory"].astype(bool, copy=True)
        net.eligibility = payload["eligibility"].astype(np.float32, copy=True)
        if "motor_w" not in payload.files or "motor_b" not in payload.files:
            raise ValueError(
                "This saved brain predates motor-decoder persistence. Rerun run_pretorius.py "
                "with this v0.2 package once to create a reloadable mature brain."
            )
        net.motor_w = payload["motor_w"].astype(np.float32, copy=True)
        net.motor_b = payload["motor_b"].astype(np.float32, copy=True)
        net.tick = int(payload["tick"][0])
        return net

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            v=self.v,
            rate=self.rate,
            bias=self.bias,
            w_data=self.W.data,
            w_indices=self.W.indices,
            w_indptr=self.W.indptr,
            excitatory=self.excitatory,
            eligibility=self.eligibility,
            motor_w=self.motor_w,
            motor_b=self.motor_b,
            cfg_json=np.asarray(json.dumps(self.cfg)),
            tick=np.asarray([self.tick], dtype=np.int64),
        )
