"""BC01-D3: source-anchored recurrent credit-assignment diagnostic.

Development-only. Uses BC01's existing sparse recurrent donor, existing
source cards and optional source-owned shared TF-IDF cache. No new renderer.
Every behavioral claim is compared to label-shuffled and recurrent-lesion
controls; all labels remain outside evaluation inputs.
"""
from __future__ import annotations

import copy
from hashlib import sha256
import json

import numpy as np

from biocircuit.bc01 import Corpus, make_circuit, neural_config, represent
from biocircuit.bc01_decisions import EVAL_ACTIONS, validate_cards
from persona_net.encoding import ACTIONS, ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet


def _scores(net, vector, ticks: int) -> tuple[dict, dict, np.ndarray]:
    net.reset_fast_state()
    for _ in range(ticks):
        net.step(vector, reward=0.0, learn=False)
    populations = {a: float(net.rate[net.action_populations[a]].mean())
                   for a in EVAL_ACTIONS}
    raw = np.asarray([populations[a] for a in EVAL_ACTIONS], dtype=np.float64)
    probabilities = np.exp(30.0 * (raw - raw.max()))
    probabilities /= probabilities.sum()
    scores = {a: float(v) for a, v in zip(EVAL_ACTIONS, probabilities)}
    trace = {
        "population_mean": populations,
        "mean_rate": float(net.rate.mean()),
        "rate_std": float(net.rate.std()),
        "rate_min": float(net.rate.min()),
        "rate_max": float(net.rate.max()),
        "near_zero_fraction": float(np.mean(net.rate < 0.01)),
        "near_one_fraction": float(np.mean(net.rate > 0.99)),
        "bias_mean": float(net.bias.mean()),
        "bias_std": float(net.bias.std()),
    }
    return scores, trace, net.rate.copy()


def _curriculum(net, corpus, encoder, circuit, shared, ticks: int):
    for item in corpus.records:
        x = represent(item["memory_text"], encoder, circuit,
                      shared=shared, event_id=item["event_id"])
        net.reset_fast_state()
        for _ in range(ticks):
            net.step(x, reward=0.0, learn=True)


def _train_generic(net, cards, encoder, circuit, shared, epochs: int,
                   ticks: int, labels, plastic: bool = True):
    for _ in range(epochs):
        for card, label in zip(cards, labels):
            x = represent(card["training_cues"], encoder, circuit, shared=shared)
            x[encoder.action_offset + ACTIONS.index(label)] = 1.0
            net.reset_fast_state()
            for _ in range(ticks):
                net.step(x, reward=1.0, learn=plastic)


def _train_targeted(net, cards, encoder, circuit, shared, epochs: int,
                    ticks: int, labels, eta: float, blank_cues: bool = False):
    """One bounded three-factor update per event on EXISTING sparse W entries.

    Factor 1: cue-only presynaptic firing above target (context).
    Factor 2: target action post response caused by separate teaching input.
    Factor 3: source-anchored target-population eligibility (class label).

    The action channel is NEVER presented to evaluation probes. Recurrent
    topology and Dale sign are preserved; motor decoder never trained.
    """
    if ticks < 2:
        raise ValueError("D3 requires >= 2 ticks for paired cue/teacher phases")
    pre_ticks = ticks // 2
    post_ticks = ticks - pre_ticks
    masks = {a: np.isin(net.post_idx, net.action_populations[a]) for a in EVAL_ACTIONS}
    for _ in range(epochs):
        for card, label in zip(cards, labels):
            stimulus = represent(card["training_cues"], encoder, circuit, shared=shared)
            if blank_cues:
                stimulus[:] = 0.0
            cue = stimulus.copy()
            net.reset_fast_state()
            for _ in range(pre_ticks):
                net.step(cue, learn=False)
            pre_response = net.rate.copy()
            # Teacher never contains event IDs, source text or policy
            # features; it activates the donor's EXISTING action channel.
            teacher = cue.copy()
            teacher[encoder.action_offset + ACTIONS.index(label)] = 1.0
            for _ in range(post_ticks):
                net.step(teacher, learn=False)
            post_response = net.rate.copy()
            # Only the taught action's existing incoming recurrent edges
            # receive activity-gated weight adjustments.
            correlation = (
                np.maximum(pre_response[net.pre_idx] - net.target_rate, 0.0)
                * np.maximum(post_response[net.post_idx] - pre_response[net.post_idx], 0.0)
                * masks[label]
            )
            net.W.data += (eta * correlation).astype(np.float32)
            sign_exc = net.excitatory[net.pre_idx]
            net.W.data[sign_exc] = np.clip(net.W.data[sign_exc], 0.0, net.max_abs_weight)
            net.W.data[~sign_exc] = np.clip(net.W.data[~sign_exc], -net.max_abs_weight, 0.0)


def _distribution(rows, key):
    choices = [r["conditions"][key]["choice"] for r in rows]
    counts = {a: choices.count(a) for a in EVAL_ACTIONS}
    hits = sum(r["conditions"][key]["choice"] == r["target"] for r in rows)
    return {"accuracy": hits / len(rows), "choice_histogram": counts,
            "distinct_choices": sum(v > 0 for v in counts.values()),
            "constant_action": max(counts.values()) == len(rows),
            "confusion": {
                actual: {guess: sum(r["target"] == actual and
                                    r["conditions"][key]["choice"] == guess
                                    for r in rows)
                         for guess in EVAL_ACTIONS}
                for actual in EVAL_ACTIONS
            }}


def experiment(corpus: Corpus, cards: tuple[dict, ...], *,
               neurons=256, seed=31, mode="local", shared=None,
               background_ticks=8, epochs=3, card_ticks=32,
               settle_ticks=32, eta=0.03, gain=1.0) -> dict:
    cards = validate_cards(corpus, cards)
    if shared is not None:
        shared.verify_corpus(corpus)
    if mode not in ("local", "global", "generic") or min(
        background_ticks, epochs, card_ticks, settle_ticks
    ) < 1 or card_ticks < 2:
        raise ValueError("Invalid D3 budget or topology")
    if not np.isfinite(eta) or eta <= 0 or not np.isfinite(gain) or gain <= 0:
        raise ValueError("Invalid D3 learning multiplier")
    encoder = ExperienceEncoder(sensory_dim=256)
    circuit = make_circuit(neurons, seed, mode)
    cfg = neural_config(neurons, seed)
    cfg["hebb_lr"] *= gain
    cfg["reward_lr"] *= gain
    net = PlasticRecurrentPersonaNet(cfg, encoder)
    _curriculum(net, corpus, encoder, circuit, shared, background_ticks)
    original_weights = net.W.data.copy()
    original_motor = net.motor_w.copy()
    original_motor_bias = net.motor_b.copy()
    labels = tuple(c["action"] for c in cards)
    rng = np.random.default_rng(seed + 1977)
    shuffled = list(labels)
    # D1 already used this same deterministic label derangement.
    for _ in range(1000):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(labels, shuffled)):
            break
    else:
        raise ValueError("Insufficient action diversity for matched label shuffle")
    models = {name: copy.deepcopy(net) for name in (
        "generic_hebb", "targeted", "targeted_shuffled",
        "targeted_blank_cue", "no_training"
    )}
    _train_generic(models["generic_hebb"], cards, encoder, circuit, shared,
                   epochs, card_ticks, labels)
    _train_targeted(models["targeted"], cards, encoder, circuit, shared,
                    epochs, card_ticks, labels, eta)
    _train_targeted(models["targeted_shuffled"], cards, encoder, circuit, shared,
                    epochs, card_ticks, shuffled, eta)
    _train_targeted(models["targeted_blank_cue"], cards, encoder, circuit, shared,
                    epochs, card_ticks, labels, eta, blank_cues=True)
    models["targeted_recurrent_lesion"] = copy.deepcopy(models["targeted"])
    models["targeted_recurrent_lesion"].W.data[:] = original_weights
    if any(not np.array_equal(model.motor_w, original_motor)
           or not np.array_equal(model.motor_b, original_motor_bias)
           for model in models.values()):
        raise AssertionError("D3 must never train a motor decoder")
    query_vectors = [represent(card["probe"], encoder, circuit, shared=shared)
                     for card in cards]
    # Pre-/post-learning physiology: use fixed query order and independent
    # fast-state reset for every probe.
    rows = []
    activities = {name: [] for name in models}
    for card, query in zip(cards, query_vectors):
        conditions = {}
        for name, trained in models.items():
            score, physiology, activity = _scores(trained, query, settle_ticks)
            activities[name].append(activity)
            conditions[name] = {
                "choice": max(EVAL_ACTIONS, key=lambda a: score[a]),
                "scores": score, "trace": physiology,
            }
        rows.append({"event_id": card["event_id"], "target": card["action"],
                     "probe": card["probe"], "conditions": conditions})
    metrics = {name: _distribution(rows, name) for name in models}
    spread = {}
    physiology = {}
    for name, sequence in activities.items():
        stack = np.stack(sequence)
        # Normalized RMS across cues measures neural differentiation, not
        # autobiographical correctness.
        spread[name] = float(np.sqrt(np.mean((stack - stack.mean(axis=0)) ** 2)))
        physiology[name] = {
            "mean_rate": float(stack.mean()),
            "std_rate": float(stack.std()),
            "fraction_below_point01": float(np.mean(stack < 0.01)),
            "fraction_above_point99": float(np.mean(stack > 0.99)),
            "bias_mean": float(models[name].bias.mean()),
            "bias_std": float(models[name].bias.std()),
        }
    deltas = {name: float(np.abs(model.W.data - original_weights).sum())
              for name, model in models.items()}
    if any(not np.isfinite(v) for v in list(spread.values()) + list(deltas.values())):
        raise ValueError("D3 numerical instability")
    # A score gain alone is insufficient: require content dependence and a
    # specific advantage over lesion, shuffled and blank-cue training.
    t = metrics["targeted"]
    gates = {
        "more_than_one_action": t["distinct_choices"] > 1,
        "above_balanced_chance": t["accuracy"] > 0.25,
        "beats_recurrent_lesion": t["accuracy"] > metrics["targeted_recurrent_lesion"]["accuracy"],
        "beats_shuffled_teacher": t["accuracy"] > metrics["targeted_shuffled"]["accuracy"],
        "beats_blank_cue_training": t["accuracy"] > metrics["targeted_blank_cue"]["accuracy"],
        "beats_generic_hebb": t["accuracy"] > metrics["generic_hebb"]["accuracy"],
    }
    return {
        "schema": "BC01-D3-development-diagnostic-v1",
        "source_commit": corpus.blob_sha,
        "annotated_cards": len(cards),
        "seed": seed, "mode": mode, "neurons": neurons,
        "shared_feature_manifest_sha256": (shared.cache_file_sha if shared else None),
        "input_encoder": ("source-owned shared TF-IDF + BC01 L3 v1"
                          if shared else "legacy BC01 hashed lexical"),
        "eta": eta, "gain": gain, "background_ticks": background_ticks,
        "card_ticks": card_ticks, "epochs": epochs, "settle_ticks": settle_ticks,
        "training_budget": "equal 32 ticks per card; targeted 16 cue + 16 teacher; generic 32 co-presented",
        "result": metrics, "neural_cue_rms_spread": spread, "physiology": physiology,
        "recurrent_delta_l1": deltas, "gates": gates,
        "passes_all_development_gates": all(gates.values()),
        "notes": "Provisional labels, reused lexical development prompts; positive outcome is not independent generalization.",
        "rows": rows,
    }
