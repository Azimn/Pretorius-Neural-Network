"""BC01-D4: sensory-versus-teaching drive scaling of the D3 network.

Development-only. Uses BC01's existing sparse recurrent donor, existing
source cards and optional source-owned shared TF-IDF cache. No new renderer.
Every behavioral claim is compared to label-shuffled and recurrent-lesion
controls; all labels remain outside evaluation inputs.
"""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile

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
                   ticks: int, labels, plastic: bool = True, sensory_gain: float = 1.0):
    for _ in range(epochs):
        for card, label in zip(cards, labels):
            x = represent(card["training_cues"], encoder, circuit, shared=shared)
            x[:encoder.sensory_dim] *= sensory_gain
            x[encoder.action_offset + ACTIONS.index(label)] = 1.0
            net.reset_fast_state()
            for _ in range(ticks):
                net.step(x, reward=1.0, learn=plastic)


def _train_targeted(net, cards, encoder, circuit, shared, epochs: int,
                    ticks: int, labels, eta: float, blank_cues: bool = False,
                    sensory_gain: float = 1.0, observer=None,
                    presynaptic_mode: str = "target_rate"):
    """One bounded three-factor update per event on EXISTING sparse W entries.

    Factor 1: cue-only presynaptic firing above target (context).
    Factor 2: target action post response caused by separate teaching input.
    Factor 3: source-anchored target-population eligibility (class label).

    The action channel is NEVER presented to evaluation probes. Recurrent
    topology and Dale sign are preserved; motor decoder never trained.
    """
    if ticks < 2:
        raise ValueError("D3 requires >= 2 ticks for paired cue/teacher phases")
    if presynaptic_mode not in ("target_rate", "cue_minus_blank"):
        raise ValueError("Unregistered D5B presynaptic eligibility mode")
    pre_ticks = ticks // 2
    post_ticks = ticks - pre_ticks
    masks = {a: np.isin(net.post_idx, net.action_populations[a]) for a in EVAL_ACTIONS}
    for epoch in range(epochs):
        for card, label in zip(cards, labels):
            stimulus = represent(card["training_cues"], encoder, circuit, shared=shared)
            stimulus[:encoder.sensory_dim] *= sensory_gain
            if blank_cues:
                stimulus[:] = 0.0
            cue = stimulus.copy()
            net.reset_fast_state()
            for _ in range(pre_ticks):
                net.step(cue, learn=False)
            pre_response = net.rate.copy()
            # D5B controlled candidate: source-specific presynaptic drive,
            # measured versus a time-matched blank first-half exposure.
            # Restore the original cue branch exactly before proceeding.
            # Default D4 mode does no extra steps and remains bit-identical.
            if presynaptic_mode == "cue_minus_blank":
                cue_v, cue_rate, cue_tick = net.v.copy(), net.rate.copy(), net.tick
                net.reset_fast_state()
                zero = np.zeros_like(cue)
                for _ in range(pre_ticks):
                    net.step(zero, learn=False)
                blank_pre_response = net.rate.copy()
                net.v[:] = cue_v
                net.rate[:] = cue_rate
                net.tick = cue_tick
                pre_factor = np.maximum(pre_response - blank_pre_response, 0.0)
            else:
                pre_factor = np.maximum(pre_response - net.target_rate, 0.0)
            # Strict paired counterfactual: run the same second-phase time
            # window *without* teaching, from the same fast state. Comparing
            # unpaired timepoints can yield a zero learning signal simply
            # because the rate network is settling down after reset.
            before_v, before_rate, before_tick = net.v.copy(), net.rate.copy(), net.tick
            for _ in range(post_ticks):
                net.step(cue, learn=False)
            no_teacher_response = net.rate.copy()
            net.v[:] = before_v
            net.rate[:] = before_rate
            net.tick = before_tick
            # Teacher never contains event IDs, source text or policy
            # features; it activates the donor's EXISTING action channel.
            teacher = cue.copy()
            teacher[encoder.action_offset + ACTIONS.index(label)] = 1.0
            for _ in range(post_ticks):
                net.step(teacher, learn=False)
            post_response = net.rate.copy()
            # Match the exact time window, not pre-/post-settling rates.
            # Counterfactual branch adds extra inference ticks and is not
            # a strict FLOP-matched comparator to global Hebbian exposure.
            correlation = (
                pre_factor[net.pre_idx]
                * np.maximum(post_response[net.post_idx] - no_teacher_response[net.post_idx], 0.0)
                * masks[label]
            )
            # D5 diagnostic observer is strictly optional and never
            # changes the original D4 update. Capture a copy ONLY for
            # audit mode, so ordinary D4 retains identical numeric behavior.
            if observer is not None:
                w_before = net.W.data.copy()
            net.W.data += (eta * correlation).astype(np.float32)
            sign_exc = net.excitatory[net.pre_idx]
            if observer is not None:
                proposed = net.W.data.copy()
            net.W.data[sign_exc] = np.clip(net.W.data[sign_exc], 0.0, net.max_abs_weight)
            net.W.data[~sign_exc] = np.clip(net.W.data[~sign_exc], -net.max_abs_weight, 0.0)
            if observer is not None:
                selected = masks[label]
                eligible = selected & (correlation > 0)
                delta = net.W.data - w_before
                fixed_cue = float(np.linalg.norm(net.Win.dot(cue)))
                teacher_col = encoder.action_offset + ACTIONS.index(label)
                fixed_teacher = float(np.linalg.norm(net.Win[:, teacher_col].toarray()))
                observer({
                    "event_id": card["event_id"], "epoch": epoch, "teaching_action": label,
                    "blank_cues": bool(blank_cues), "sensory_gain": float(sensory_gain),
                    "action_population_size": int(len(net.action_populations[label])),
                    "candidate_incoming_edges": int(np.count_nonzero(selected)),
                    "eligible_edges": int(np.count_nonzero(eligible)),
                    "eligible_exc": int(np.count_nonzero(eligible & sign_exc)),
                    "eligible_inh": int(np.count_nonzero(eligible & ~sign_exc)),
                    "modified_edges": int(np.count_nonzero(delta)),
                    "clipped_edges": int(np.count_nonzero(eligible & (proposed != net.W.data))),
                    "clipped_dale_edges": int(np.count_nonzero(
                        eligible & (((proposed < 0) & sign_exc) |
                                    ((proposed > 0) & ~sign_exc))
                    )),
                    "clipped_max_edges": int(np.count_nonzero(
                        eligible & (np.abs(proposed) > net.max_abs_weight)
                    )),
                    "cue_pre_rate_mean": float(np.mean(pre_response)),
                    "cue_pre_above_target_fraction": float(np.mean(pre_response > net.target_rate)),
                    "teacher_counterfactual_target_mean": float(np.mean(
                        (post_response - no_teacher_response)[net.action_populations[label]]
                    )),
                    "teacher_counterfactual_target_positive_fraction": float(np.mean(
                        (post_response - no_teacher_response)[net.action_populations[label]] > 0
                    )),
                    "eligibility_sum": float(np.sum(correlation)),
                    "eligibility_max": float(np.max(correlation)) if correlation.size else 0.0,
                    "recurrent_delta_l1": float(np.sum(np.abs(delta))),
                    "exc_recurrent_delta_l1": float(np.sum(np.abs(delta[sign_exc]))),
                    "inh_recurrent_delta_l1": float(np.sum(np.abs(delta[~sign_exc]))),
                    "fixed_cue_drive_l2": fixed_cue,
                    "fixed_teacher_drive_l2": fixed_teacher,
                    "cue_teacher_input_ratio": fixed_cue / max(fixed_teacher, 1e-12),
                    "bias_mean": float(np.mean(net.bias)),
                    "mean_population_rate": float(np.mean(net.rate[net.action_populations[label]])),
                })


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
               settle_ticks=32, eta=0.03, gain=1.0,
               sensory_gain=1.0, checkpoint_dir=None,
               presynaptic_mode="target_rate", diagnostic_observer=None) -> dict:
    cards = validate_cards(corpus, cards)
    if shared is not None:
        shared.verify_corpus(corpus)
    if mode not in ("local", "global", "generic") or min(
        background_ticks, epochs, card_ticks, settle_ticks
    ) < 1 or card_ticks < 2:
        raise ValueError("Invalid D3 budget or topology")
    if not np.isfinite(eta) or eta <= 0 or not np.isfinite(gain) or gain <= 0:
        raise ValueError("Invalid D4 learning multiplier")
    if presynaptic_mode not in ("target_rate", "cue_minus_blank"):
        raise ValueError("Unregistered D5B presynaptic learning rule")
    if not np.isfinite(sensory_gain) or sensory_gain <= 0 or sensory_gain > 12.0:
        raise ValueError("D4 sensory gain must be finite and in (0,12]")
    # The same 450-memory background curriculum is used for every gain:
    # ONLY source-card learning and querying change. No refit, no
    # plasticity coefficient adjustment, no action-teacher scaling.
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
                   epochs, card_ticks, labels, sensory_gain=sensory_gain)
    _train_targeted(models["targeted"], cards, encoder, circuit, shared,
                    epochs, card_ticks, labels, eta, sensory_gain=sensory_gain,
                    presynaptic_mode=presynaptic_mode,
                    observer=diagnostic_observer)
    _train_targeted(models["targeted_shuffled"], cards, encoder, circuit, shared,
                    epochs, card_ticks, shuffled, eta, sensory_gain=sensory_gain,
                    presynaptic_mode=presynaptic_mode)
    _train_targeted(models["targeted_blank_cue"], cards, encoder, circuit, shared,
                    epochs, card_ticks, labels, eta, blank_cues=True,
                    sensory_gain=sensory_gain,
                    presynaptic_mode=presynaptic_mode)
    models["targeted_recurrent_lesion"] = copy.deepcopy(models["targeted"])
    models["targeted_recurrent_lesion"].W.data[:] = original_weights
    if any(not np.array_equal(model.motor_w, original_motor)
           or not np.array_equal(model.motor_b, original_motor_bias)
           for model in models.values()):
        raise AssertionError("D3 must never train a motor decoder")
    if checkpoint_dir is None:
        scratch = tempfile.TemporaryDirectory(prefix="bc01_d3_")
        checkpoint = Path(scratch.name) / "targeted.npz"
    else:
        scratch = None
        checkpoint = Path(checkpoint_dir) / f"bc01_d4_gain{sensory_gain:g}_{mode}_{seed}.npz"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    models["targeted"].save(checkpoint)
    restored = PlasticRecurrentPersonaNet.load(checkpoint, cfg, encoder)
    query_vectors = [represent(card["probe"], encoder, circuit, shared=shared)
                     for card in cards]
    # No change to externally fitted TF-IDF features or action channels:
    # only the existing 256 sensory amplitudes are multiplied at inference.
    for query in query_vectors:
        query[:encoder.sensory_dim] *= sensory_gain
        if np.count_nonzero(query[encoder.action_offset:]):
            raise AssertionError("Action/label leakage in query input")
    # Quantify whether the fixed input fan-in makes narrative cues weak
    # relative to the action-teaching injection. Teaching drives are measured
    # OFFLINE for diagnosis and NEVER passed into query inference.
    coupling = []
    for card, query in zip(cards, query_vectors):
        sensory_norm = float(np.linalg.norm(net.Win.dot(query)))
        action_col = encoder.action_offset + ACTIONS.index(card["action"])
        teaching_norm = float(np.linalg.norm(net.Win[:, action_col].toarray()))
        coupling.append({
            "event_id": card["event_id"],
            "sensory_drive_l2": sensory_norm,
            "teacher_drive_l2": teaching_norm,
            "sensory_to_teacher_drive_ratio": sensory_norm / max(teaching_norm, 1e-12),
        })
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
    restart_equal = []
    for query, row in zip(query_vectors, rows):
        scores, _, _ = _scores(restored, query, settle_ticks)
        restart_equal.append(scores == row["conditions"]["targeted"]["scores"])
    if scratch is not None:
        scratch.cleanup()
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
        "schema": ("BC01-D5B-cue-minus-blank-development-v1"
                   if presynaptic_mode == "cue_minus_blank"
                   else "BC01-D4-source-input-amplitude-v1"),
        "source_commit": corpus.blob_sha,
        "annotated_cards": len(cards),
        "seed": seed, "mode": mode, "neurons": neurons,
        "shared_feature_manifest_sha256": (shared.cache_file_sha if shared else None),
        "input_encoder": ("source-owned shared TF-IDF + BC01 L3 v1"
                          if shared else "legacy BC01 hashed lexical"),
        "eta": eta, "gain": gain, "sensory_gain": sensory_gain,
        "background_sensory_gain": 1.0,
        "background_ticks": background_ticks,
        "card_ticks": card_ticks, "epochs": epochs, "settle_ticks": settle_ticks,
        "training_budget": "same 450 background exposures across gains, plus equal card teacher/cue exposure ticks; targeted additionally computes a matched no-teacher counterfactual and costs more CPU than generic",
        "input_drive": {
            "per_source_card": coupling,
            "sensory_norm_mean": float(np.mean([x["sensory_drive_l2"] for x in coupling])),
            "teacher_norm_mean": float(np.mean([x["teacher_drive_l2"] for x in coupling])),
            "sensory_to_teacher_ratio_mean": float(np.mean([x["sensory_to_teacher_drive_ratio"] for x in coupling])),
            "interpretation": "OFFLINE reference target teacher column; never injected into evaluation input",
        },
        "result": metrics, "neural_cue_rms_spread": spread, "physiology": physiology,
        "recurrent_delta_l1": deltas,
        "checkpoint_restart_exact": bool(all(restart_equal)),
        "checkpoint_name": checkpoint.name if checkpoint_dir is not None else None,
        "gates": gates,
        "passes_all_development_gates": all(gates.values()),
        "notes": ("D5B prespecified cue-vs-matched-blank presynaptic eligibility only; extra blank branch costs CPU. Reused development source cards are not independent."
                  if presynaptic_mode == "cue_minus_blank" else
                  "Posthoc prespecified gains 1,4,12 on card cue + query sensory channels only. Shared source TF-IDF frozen; original 450-memory background identical across gains. Provisional labels; reused development cues cannot establish generalization."),
        "rows": rows,
    }
