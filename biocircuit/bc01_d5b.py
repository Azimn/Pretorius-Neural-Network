"""BC01-D5B: precommitted cue-minus-blank eligibility comparison.

Exactly one local rule changes from D4: presynaptic eligibility uses the
matched-time cue-minus-blank rate, not a global target-rate comparison.
The motor-only control trains *output* weights, never recurrent synapses.
"""
from __future__ import annotations

import numpy as np

from biocircuit.bc01 import make_circuit, neural_config, represent
from biocircuit.bc01_decisions import EVAL_ACTIONS, motor_decoder_scores, validate_cards
from biocircuit.bc01_d4 import _curriculum, experiment as d4_experiment
from persona_net.encoding import ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet


def _motor_only(corpus, cards, *, shared, seed, mode, neurons,
                gain, epochs, card_ticks, background_ticks, settle_ticks):
    encoder = ExperienceEncoder(sensory_dim=256)
    circuit = make_circuit(neurons, seed, mode)
    net = PlasticRecurrentPersonaNet(neural_config(neurons, seed), encoder)
    _curriculum(net, corpus, encoder, circuit, shared, background_ticks)
    background_w = net.W.data.copy()
    for _ in range(epochs):
        for card in cards:
            x = represent(card["training_cues"], encoder, circuit, shared=shared)
            x[:encoder.sensory_dim] *= gain
            net.reset_fast_state()
            for _ in range(card_ticks):
                net.step(x, learn=False)
            net.learn_motor(card["action"])
    if not np.array_equal(net.W.data, background_w):
        raise AssertionError("Motor-only comparator changed learned recurrence")
    predictions = []
    for card in cards:
        query = represent(card["probe"], encoder, circuit, shared=shared)
        query[:encoder.sensory_dim] *= gain
        if np.any(query[encoder.action_offset:]):
            raise AssertionError("Motor-only action label leaked into inference")
        scores = motor_decoder_scores(net, query, settle_ticks)
        predictions.append({
            "event_id": card["event_id"],
            "target": card["action"], "scores": scores,
            "choice": max(EVAL_ACTIONS, key=scores.get),
        })
    return {
        "accuracy": sum(p["choice"] == p["target"] for p in predictions)/len(predictions),
        "distinct_choices": len(set(p["choice"] for p in predictions)),
        "w_recurrent_unchanged": bool(np.array_equal(net.W.data, background_w)),
        "motor_decoder_supervised": True,
        "rows": predictions,
    }


def comparison(corpus, cards, *, seed=31, mode="local", sensory_gain=1.,
               neurons=256, shared=None, epochs=3, card_ticks=32,
               background_ticks=8, settle_ticks=32, eta=0.03,
               checkpoint_root=None):
    cards = validate_cards(corpus, cards)
    if sensory_gain not in (1.,12.):
        raise ValueError("D5B only prespecified gains 1 and 12")
    if shared is not None:
        shared.verify_corpus(corpus)
    settings = dict(seed=seed, mode=mode, sensory_gain=sensory_gain,
                    neurons=neurons, shared=shared, epochs=epochs,
                    card_ticks=card_ticks, background_ticks=background_ticks,
                    settle_ticks=settle_ticks, eta=eta)
    if checkpoint_root:
        from pathlib import Path
        base_root = Path(checkpoint_root)
        original_dir = base_root / "legacy"
        corrected_dir = base_root / "corrected"
    else:
        original_dir = corrected_dir = None

    legacy = d4_experiment(corpus, cards, **settings,
                           presynaptic_mode="target_rate",
                           checkpoint_dir=original_dir)
    targeted_traces = []
    corrected = d4_experiment(corpus, cards, **settings,
                              presynaptic_mode="cue_minus_blank",
                              checkpoint_dir=corrected_dir,
                              diagnostic_observer=targeted_traces.append)
    if len(targeted_traces) != len(cards) * epochs:
        raise AssertionError("Missing source-anchored corrected training events")
    if not all([legacy["checkpoint_restart_exact"],
                corrected["checkpoint_restart_exact"]]):
        raise ValueError("One neural learning checkpoint failed exact replay")
    # Explicitly prove that only the recurrent targeted rule changes:
    # background-only and generic full Hebbian reference outcomes identical.
    for control in ("no_training", "generic_hebb"):
        if legacy["result"][control] != corrected["result"][control]:
            raise AssertionError("D5B changed a baseline learning control")
        if any(a["conditions"][control]["scores"] != b["conditions"][control]["scores"]
               for a,b in zip(legacy["rows"], corrected["rows"])):
            raise AssertionError("D5B changed a reference neural policy")
    motor = _motor_only(
        corpus, cards, shared=shared, seed=seed, mode=mode, neurons=neurons,
        gain=sensory_gain, epochs=epochs, card_ticks=card_ticks,
        background_ticks=background_ticks, settle_ticks=settle_ticks,
    )
    controls = (
        "targeted_recurrent_lesion", "targeted_shuffled",
        "targeted_blank_cue", "no_training", "generic_hebb",
    )
    targeted = corrected["result"]["targeted"]
    strict_controls = {name: targeted["accuracy"] >
                       corrected["result"][name]["accuracy"] for name in controls}
    strict_controls["more_than_one_predicted_action"] = targeted["distinct_choices"] > 1
    strict_controls["above_balanced_chance"] = targeted["accuracy"] > 0.25
    per_card = [{
        "event_id": source["event_id"], "target": source["target"],
        "original": source["conditions"]["intact"] if "intact" in source["conditions"] else source["conditions"]["targeted"],
        "cue_contrast": trial["conditions"]["targeted"],
        "cue_contrast_recurrent_lesion": trial["conditions"]["targeted_recurrent_lesion"],
        "cue_contrast_shuffled": trial["conditions"]["targeted_shuffled"],
        "cue_contrast_blank": trial["conditions"]["targeted_blank_cue"],
    } for source,trial in zip(legacy["rows"], corrected["rows"])]
    return {
        "schema": "BC01-D5B-presynaptic-cue-contrast-v1",
        "source_git_blob": corpus.blob_sha,
        "shared_feature_manifest_sha256": shared.cache_file_sha if shared else None,
        "seed": seed, "mode": mode, "sensory_gain": sensory_gain,
        "source_card_count": len(cards), "neurons": neurons,
        "original_D4": {
            "accuracy": legacy["accuracy"] if "accuracy" in legacy else legacy["result"],
            "recurrent_delta_l1": legacy["recurrent_delta_l1"],
            "checkpoint_exact": legacy["checkpoint_restart_exact"],
        },
        "cue_contrast": {
            "accuracy": {name: row["accuracy"] for name,row in corrected["result"].items()},
            "choice_histogram": {name: row["choice_histogram"] for name,row in corrected["result"].items()},
            "recurrent_delta_l1": corrected["recurrent_delta_l1"],
            "checkpoint_exact": corrected["checkpoint_restart_exact"],
            "neural_cue_rms_spread": corrected["neural_cue_rms_spread"],
            "mean_neural_activity": corrected["physiology"],
        },
        "motor_decoder_only": motor,
        "cue_evoked_eligibility": {
            "presentations": len(targeted_traces),
            "eligible_edges_mean": float(np.mean([x["eligible_edges"] for x in targeted_traces])),
            "candidate_edges_mean": float(np.mean([x["candidate_incoming_edges"] for x in targeted_traces])),
            "modified_edges_mean": float(np.mean([x["modified_edges"] for x in targeted_traces])),
            "mean_weight_delta_l1": float(np.mean([x["recurrent_delta_l1"] for x in targeted_traces])),
            "eligible_fraction_mean": float(np.mean([
                x["eligible_edges"] / max(1,x["candidate_incoming_edges"])
                for x in targeted_traces
            ])),
            "per_event": targeted_traces,
        },
        "strict_development_gates": strict_controls,
        "passes_all_gates": bool(all(strict_controls.values())),
        "per_card": per_card,
        "limitations": "Correct source labels are human interpretations on reused development cues; this single rule change does not establish independently tested autobiographical memory.",
    }
