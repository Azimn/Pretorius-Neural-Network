"""BC01-D5: exact observational audit of D4 recurrent credit assignment.

This is a diagnostic of the *existing* fixed-network update, not a replacement
neural architecture or training rule. Source action labels are used only in
training and evaluation metadata, never inference sensory features.
"""
from __future__ import annotations

import copy
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from biocircuit.bc01 import make_circuit, neural_config, represent
from biocircuit.bc01_decisions import EVAL_ACTIONS, validate_cards
from biocircuit.bc01_d4 import _curriculum, _scores, _train_targeted, experiment as d4_experiment
from persona_net.encoding import ACTIONS, ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet


def _target_margin(scores: dict[str, float], target: str) -> float:
    return float(scores[target] - max(scores[a] for a in EVAL_ACTIONS if a != target))


def _summarize(entries: list[dict]) -> dict:
    if not entries:
        raise ValueError("Missing D5 observational traces")
    metrics = (
        "candidate_incoming_edges", "eligible_edges", "modified_edges",
        "eligible_exc", "eligible_inh", "clipped_edges", "clipped_dale_edges",
        "clipped_max_edges", "cue_pre_rate_mean",
        "cue_pre_above_target_fraction", "teacher_counterfactual_target_mean",
        "teacher_counterfactual_target_positive_fraction",
        "eligibility_sum", "eligibility_max", "recurrent_delta_l1",
        "exc_recurrent_delta_l1", "inh_recurrent_delta_l1",
        "fixed_cue_drive_l2", "fixed_teacher_drive_l2",
        "cue_teacher_input_ratio", "mean_population_rate",
    )
    summary = {
        "observations": len(entries),
        "mean": {key: float(np.mean([x[key] for x in entries])) for key in metrics},
        "total": {key: int(sum(x[key] for x in entries)) for key in (
            "eligible_edges", "modified_edges", "eligible_exc", "eligible_inh",
            "clipped_edges", "clipped_dale_edges", "clipped_max_edges"
        )},
        "by_action": {},
    }
    for action in EVAL_ACTIONS:
        subset = [x for x in entries if x["teaching_action"] == action]
        if not subset:
            raise ValueError("Missing one teaching class from D5")
        summary["by_action"][action] = {
            "observations": len(subset),
            "eligible_mean": float(np.mean([x["eligible_edges"] for x in subset])),
            "recurrent_delta_l1_mean": float(np.mean([x["recurrent_delta_l1"] for x in subset])),
            "teacher_response_mean": float(np.mean(
                [x["teacher_counterfactual_target_mean"] for x in subset])),
        }
    return summary


def _rate_policy(net, x, ticks: int, target: str):
    score, trace, state = _scores(net, x, ticks)
    return {"choice": max(EVAL_ACTIONS, key=lambda a: score[a]),
            "target_margin": _target_margin(score, target),
            "target_score": float(score[target]), "scores": score,
            "trace": trace, "population_activity": {
                a: float(state[net.action_populations[a]].mean()) for a in EVAL_ACTIONS
            }}


def audit(corpus, cards, *, seed=31, mode="local", sensory_gain=12.0,
          neurons=256, shared=None, epochs=3, card_ticks=32,
          background_ticks=8, settle_ticks=32, eta=0.03,
          checkpoint_dir: Path | None = None) -> dict:
    cards = validate_cards(corpus, cards)
    if shared is not None:
        shared.verify_corpus(corpus)
    if sensory_gain not in (1.0, 12.0):
        raise ValueError("D5 diagnosed gains are frozen at 1 and 12")
    if mode not in ("local", "global", "generic") or min(
        neurons, epochs, card_ticks, background_ticks, settle_ticks
    ) < 1:
        raise ValueError("Invalid D5 budget")
    encoder = ExperienceEncoder(sensory_dim=256)
    circuit = make_circuit(neurons, seed, mode)
    cfg = neural_config(neurons, seed)
    pre_card = PlasticRecurrentPersonaNet(cfg, encoder)
    _curriculum(pre_card, corpus, encoder, circuit, shared, background_ticks)
    w_original = pre_card.W.data.copy()
    original_motor = pre_card.motor_w.copy()
    original_motor_b = pre_card.motor_b.copy()
    trained = copy.deepcopy(pre_card)
    logs = []
    _train_targeted(
        trained, cards, encoder, circuit, shared,
        epochs=epochs, ticks=card_ticks,
        labels=tuple(c["action"] for c in cards),
        eta=eta, sensory_gain=sensory_gain, observer=logs.append,
    )
    if not np.array_equal(trained.motor_w, original_motor) or not np.array_equal(
        trained.motor_b, original_motor_b
    ):
        raise AssertionError("D5 must not train output decoder")

    # Keep exact D4 baseline and independently reproduce the intact model:
    # diagnostic callback must not change a single policy floating value.
    baseline = d4_experiment(
        corpus, cards, seed=seed, mode=mode, sensory_gain=sensory_gain,
        neurons=neurons, shared=shared, epochs=epochs, card_ticks=card_ticks,
        background_ticks=background_ticks, settle_ticks=settle_ticks, eta=eta,
        checkpoint_dir=checkpoint_dir,
    )
    lesioned = copy.deepcopy(trained)
    lesioned.W.data[:] = w_original
    changed = np.flatnonzero(trained.W.data != w_original)
    if not changed.size:
        max_edge = None
        single_edge = copy.deepcopy(trained)
    else:
        idx = int(changed[np.argmax(np.abs(trained.W.data[changed] - w_original[changed]))])
        max_edge = {
            "csr_data_index": idx,
            "post_neuron": int(trained.post_idx[idx]),
            "pre_neuron": int(trained.pre_idx[idx]),
            "learned_delta": float(trained.W.data[idx] - w_original[idx]),
            "pre_is_excitatory": bool(trained.excitatory[trained.pre_idx[idx]]),
            "belongs_to_action_population": [
                a for a in EVAL_ACTIONS if trained.post_idx[idx] in trained.action_populations[a]
            ],
        }
        single_edge = copy.deepcopy(trained)
        single_edge.W.data[idx] = w_original[idx]
    paired = []
    for card, base_row in zip(cards, baseline["rows"]):
        vector = represent(card["probe"], encoder, circuit, shared=shared)
        vector[:encoder.sensory_dim] *= sensory_gain
        if np.any(vector[encoder.action_offset:] != 0):
            raise AssertionError("Action teacher leaked into D5 probe")
        intact = _rate_policy(trained, vector, settle_ticks, card["action"])
        assert intact["scores"] == base_row["conditions"]["targeted"]["scores"], (
            "D5 observer changed D4 neural output"
        )
        lesion = _rate_policy(lesioned, vector, settle_ticks, card["action"])
        ablation = _rate_policy(single_edge, vector, settle_ticks, card["action"])
        blank = vector.copy()
        blank[:encoder.sensory_dim] = 0.0
        no_cue = _rate_policy(trained, blank, settle_ticks, card["action"])
        paired.append({
            "event_id": card["event_id"], "action": card["action"],
            "intact": intact, "recurrent_lesion": lesion,
            "strongest_single_edge_reversion": ablation,
            "no_cue_query": no_cue,
            "target_margin_change_from_lesion": float(
                intact["target_margin"] - lesion["target_margin"]
            ),
            "target_margin_change_from_single_edge": float(
                intact["target_margin"] - ablation["target_margin"]
            ),
            "target_margin_change_from_cue": float(
                intact["target_margin"] - no_cue["target_margin"]
            ),
        })
    summary = _summarize(logs)
    summary["distinct_training_card_ids"] = len(set(x["event_id"] for x in logs))
    summary["mean_abs_target_margin_due_to_recurrence"] = float(np.mean([
        abs(row["target_margin_change_from_lesion"]) for row in paired
    ]))
    summary["mean_abs_target_margin_due_to_single_edge"] = float(np.mean([
        abs(row["target_margin_change_from_single_edge"]) for row in paired
    ]))
    summary["mean_abs_target_margin_due_to_cue"] = float(np.mean([
        abs(row["target_margin_change_from_cue"]) for row in paired
    ]))
    summary["lesion_choice_flips"] = sum(
        row["intact"]["choice"] != row["recurrent_lesion"]["choice"] for row in paired
    )
    summary["no_cue_choice_flips"] = sum(
        row["intact"]["choice"] != row["no_cue_query"]["choice"] for row in paired
    )
    summary["targeted_recurrent_total_delta_l1"] = float(
        np.sum(np.abs(trained.W.data - w_original))
    )
    if not math.isfinite(summary["mean_abs_target_margin_due_to_recurrence"]):
        raise ValueError("Nonfinite D5 measured effects")
    return {
        "schema": "BC01-D5-observational-credit-localization-v1",
        "seed": seed, "mode": mode, "sensory_gain": sensory_gain,
        "source_git_blob": corpus.blob_sha,
        "shared_cache_manifest_sha256": shared.cache_file_sha if shared else None,
        "source_card_count": len(cards), "neurons": neurons,
        "recurrent_csr_nonzeros": int(trained.W.nnz),
        "checkpoint_restart_exact": baseline["checkpoint_restart_exact"],
        "fixed_D4_baseline_accuracy": baseline["result"]["targeted"]["accuracy"],
        "fixed_D4_baseline_gates": baseline["gates"],
        "max_abs_learned_edge": max_edge,
        "summary": summary, "training_event_traces": logs,
        "query_counterfactual_traces": paired,
        "limitations": "Repeated D1-D4 interpreted in-sample labels, no semantic or causal personality evidence; single-edge reversion is a sensitivity test, not a proof of useful learning.",
    }
