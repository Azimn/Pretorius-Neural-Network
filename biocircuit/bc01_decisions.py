"""BC01-D1: source-anchored, development-only recurrent decision assay.

No claims of source entailment, independent evaluation, or character identity.
The action labels are auditably HUMAN-INTERPRETED source decisions, not canon.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from biocircuit.bc01 import (
    Corpus, EVENT_ID, SOURCE_BLOB, SOURCE_COMMIT,
    load_corpus, make_circuit, neural_config, represent, words,
)
from persona_net.encoding import ACTIONS, ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet

CARD_PATH = Path(__file__).resolve().parents[1] / "resources/biocircuit/bc01_decision_cards_v1.json"
EVAL_ACTIONS = ("challenge", "create", "cooperate", "approach")
ALLOWED_MODES = ("local", "global", "generic")


def load_cards(corpus: Corpus, path: Path = CARD_PATH) -> tuple[dict, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "BC01 decision cards v1":
        raise ValueError("Invalid policy card schema")
    if payload.get("source_commit") != SOURCE_COMMIT or payload.get("source_git_blob") != SOURCE_BLOB:
        raise ValueError("Policy cards refer to an unpinned source")
    return validate_cards(corpus, tuple(payload["cards"]))


def validate_cards(corpus: Corpus, cards: tuple[dict, ...]) -> tuple[dict, ...]:
    if len(cards) < 2:
        raise ValueError("At least two decision cards required")
    lookup = corpus.by_id()
    used: set[str] = set()
    for card in cards:
        eid = card["event_id"]
        if eid not in lookup or eid in used:
            raise ValueError("Missing or duplicated anchored event")
        if card["action"] not in EVAL_ACTIONS:
            raise ValueError("Unknown action interpretation")
        source = lookup[eid]
        if card["source_decision"] != source["decisions"]:
            raise ValueError("Interpretation no longer matches the original source decision")
        cues = words(" ".join(source["recall_cues"]))
        training = words(card["training_cues"])
        probe = words(card["probe"])
        if not training or not probe or len(probe & cues) < 2 or len(training & cues) < 2:
            raise ValueError("Training and probe must be grounded in source recall cues")
        if EVENT_ID.search(card["probe"]) or EVENT_ID.search(card["training_cues"]):
            raise ValueError("Event IDs cannot appear in neural stimuli")
        if card["action"] in words(card["probe"]):
            raise ValueError("Action leakage in probe")
        used.add(eid)
    return cards


def read_population(net: PlasticRecurrentPersonaNet, x: np.ndarray, ticks: int) -> dict[str, float]:
    """Fixed action populations, no learned supervised motor decoder."""
    if ticks < 1:
        raise ValueError("Invalid settling ticks")
    net.reset_fast_state()
    for _ in range(ticks):
        net.step(x, learn=False)
    raw = np.asarray(
        [net.rate[net.action_populations[action]].mean() for action in EVAL_ACTIONS],
        dtype=np.float64,
    )
    # Relative rate rather than a separately trainable decoder.
    raw -= raw.max()
    probs = np.exp(raw * 30.0)
    probs /= probs.sum()
    return {action: float(p) for action, p in zip(EVAL_ACTIONS, probs)}


def _choice(scores: dict[str, float]) -> str:
    return max(EVAL_ACTIONS, key=lambda a: scores[a])


def _expose(net: PlasticRecurrentPersonaNet, x: np.ndarray, ticks: int,
            plastic: bool, reward: float) -> None:
    net.reset_fast_state()
    for _ in range(ticks):
        net.step(x, reward=reward, learn=plastic)


def motor_decoder_scores(net: PlasticRecurrentPersonaNet, x: np.ndarray,
                         ticks: int) -> dict[str, float]:
    """Learned output-only comparator, all recurrent parameters fixed."""
    net.reset_fast_state()
    for _ in range(ticks):
        net.step(x, learn=False)
    # Keep the same target choice set, even though motor has ten outputs.
    raw = net.action_scores()
    total = sum(raw[a] for a in EVAL_ACTIONS)
    return {a: raw[a] / total for a in EVAL_ACTIONS}


def train_decoder_only(net: PlasticRecurrentPersonaNet, cards: tuple[dict, ...],
                       encoder: ExperienceEncoder, circuit, epochs: int,
                       settle_ticks: int) -> None:
    for _ in range(epochs):
        for card in cards:
            # No action teaching channel; this control exclusively trains
            # the motor_w / motor_b decoder with a source-annotated target.
            cue = represent(card["training_cues"], encoder, circuit)
            motor_decoder_scores(net, cue, settle_ticks)
            net.learn_motor(card["action"])


def _fit_background(net: PlasticRecurrentPersonaNet, corpus: Corpus,
                    encoder: ExperienceEncoder, circuit, background_ticks: int) -> None:
    for memory in corpus.records:
        x = represent(memory["memory_text"], encoder, circuit)
        _expose(net, x, background_ticks, plastic=True, reward=0.0)


def _fit_cards(net: PlasticRecurrentPersonaNet, cards: tuple[dict, ...],
               encoder: ExperienceEncoder, circuit, epochs: int, ticks: int,
               plastic: bool = True, labels: tuple[str, ...] | None = None,
               teach: bool = True) -> None:
    for _ in range(epochs):
        for k, card in enumerate(cards):
            x = represent(card["training_cues"], encoder, circuit)
            if teach:
                label = labels[k] if labels is not None else card["action"]
                x[encoder.action_offset + ACTIONS.index(label)] = 1.0
            # Reward=+1 is an experimental association gate, not a
            # historical reward or a judgment about Pretorius's experience.
            _expose(net, x, ticks, plastic=plastic, reward=1.0)


def benchmark(corpus: Corpus, cards: tuple[dict, ...], mode: str = "local",
              neurons: int = 256, seed: int = 31, epochs: int = 3,
              card_ticks: int = 32, background_ticks: int = 8,
              settle_ticks: int = 32, plasticity_gain: float = 1.0,
              checkpoint_dir: Path | None = None) -> dict:
    """One predefined development assay; match input/budget within a topology.

    `shuffled` permutes *learning labels* only, never labels at inference.
    Lesion reverts recurrent W to pre-card state while retaining adapted bias,
    all fixed input projections and the unchanged/untrained motor decoder.
    """
    if mode not in ALLOWED_MODES:
        raise ValueError("Unrecognized topology")
    if min(epochs, card_ticks, background_ticks, settle_ticks) < 1:
        raise ValueError("Learning and probe budgets must be positive")
    if not np.isfinite(plasticity_gain) or plasticity_gain <= 0:
        raise ValueError("Invalid plasticity multiplier")
    cards = validate_cards(corpus, cards)
    encoder = ExperienceEncoder(sensory_dim=256)
    circuit = make_circuit(neurons, seed, mode)
    cfg = neural_config(neurons, seed)
    cfg["hebb_lr"] *= plasticity_gain
    cfg["reward_lr"] *= plasticity_gain
    base = PlasticRecurrentPersonaNet(cfg, encoder)
    _fit_background(base, corpus, encoder, circuit, background_ticks)
    # The shared BASE already contains identical unlabelled autobiography.
    pre_card_w = base.W.data.copy()
    trained = copy.deepcopy(base)
    no_plastic = copy.deepcopy(base)
    shuffled = copy.deepcopy(base)
    cue_only = copy.deepcopy(base)
    labels = tuple(card["action"] for card in cards)
    perm_rng = np.random.default_rng(seed + 1977)
    # A derangement is used to force a wrong association for every card.
    candidates = list(labels)
    for _ in range(1000):
        perm_rng.shuffle(candidates)
        if all(a != b for a, b in zip(labels, candidates)):
            break
    else:
        # Some smoke sets cannot derange if only one label is present.
        raise ValueError("Insufficient action diversity for shuffled control")
    _fit_cards(trained, cards, encoder, circuit, epochs, card_ticks)
    _fit_cards(no_plastic, cards, encoder, circuit, epochs, card_ticks, plastic=False)
    _fit_cards(shuffled, cards, encoder, circuit, epochs, card_ticks, labels=tuple(candidates))
    _fit_cards(cue_only, cards, encoder, circuit, epochs, card_ticks, teach=False)
    ablated = copy.deepcopy(trained)
    ablated.W.data[:] = pre_card_w
    # Decoder-only comparator: train ONLY motor output weights; no
    # recurrent weights or biases update after the common source prelude.
    decoder_only = copy.deepcopy(base)
    train_decoder_only(decoder_only, cards, encoder, circuit, epochs, settle_ticks)
    if checkpoint_dir is None:
        local = tempfile.TemporaryDirectory(prefix="bc01_decisions_")
        checkpoint_path = Path(local.name) / "trained.npz"
    else:
        local = None
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"bc01_d1_{mode}_{seed}.npz"
    trained.save(checkpoint_path)
    restored = PlasticRecurrentPersonaNet.load(checkpoint_path, cfg, encoder)
    rows = []
    for card in cards:
        query = represent(card["probe"], encoder, circuit)
        conditions = {}
        for key, model in (
            ("intact", trained), ("recurrent_lesion", ablated),
            ("no_plasticity", no_plastic), ("shuffled_labels", shuffled),
            ("cue_only", cue_only), ("restarted", restored),
        ):
            scores = read_population(model, query, settle_ticks)
            conditions[key] = {"choice": _choice(scores), "scores": scores}
        decoder_scores = motor_decoder_scores(decoder_only, query, settle_ticks)
        conditions["decoder_only"] = {"choice": _choice(decoder_scores), "scores": decoder_scores}
        # Lexical nearest-neighbour policy, uses EXTERNAL card labels:
        # this is an information-rich retrieval-only control.
        query_words = words(card["probe"])
        matches = [
            (len(query_words & words(other["training_cues"])),
             other["action"], other["event_id"]) for other in cards
        ]
        best = sorted(matches, key=lambda x: (-x[0], x[2]))[0]
        conditions["retrieval_only"] = {"choice": best[1], "lexical_overlap": best[0]}
        rows.append({
            "event_id": card["event_id"], "target": card["action"],
            "probe": card["probe"], "conditions": conditions,
            "restart_exact": conditions["intact"]["scores"] == conditions["restarted"]["scores"],
            "recurrent_lesion_changes_choice": conditions["intact"]["choice"] != conditions["recurrent_lesion"]["choice"],
            "recurrent_delta_score_max_abs": max(
                abs(conditions["intact"]["scores"][a] - conditions["recurrent_lesion"]["scores"][a])
                for a in EVAL_ACTIONS
            ),
        })
    if local is not None:
        local.cleanup()
    by_condition = {}
    for key in ("intact", "recurrent_lesion", "no_plasticity", "shuffled_labels",
                "cue_only", "decoder_only", "retrieval_only"):
        by_condition[key] = float(sum(r["conditions"][key]["choice"] == r["target"] for r in rows) / len(rows))
    counts = {a: sum(card["action"] == a for card in cards) for a in EVAL_ACTIONS}
    return {
        "schema": "BC01-D1-exploratory-v1",
        "source_commit": SOURCE_COMMIT, "source_git_blob": corpus.blob_sha,
        "source_records_exposed": len(corpus.records), "annotated_decision_cards": len(cards),
        "topology": mode, "neurons": neurons, "seed": seed,
        "encoding": "same deterministic lexical hash; no semantic model",
        "readout": "fixed untrained action-population rate; not a motor decoder",
        "plasticity_gain": plasticity_gain,
        "budgets": {"background_ticks_per_memory": background_ticks,
                    "epochs": epochs, "card_ticks_per_epoch": card_ticks,
                    "settle_ticks": settle_ticks},
        "target_histogram": counts,
        "accuracy": by_condition,
        "recurrent_lesion_choice_flips": sum(x["recurrent_lesion_changes_choice"] for x in rows),
        "mean_recurrent_lesion_score_max_abs": float(np.mean([x["recurrent_delta_score_max_abs"] for x in rows])),
        "recurrent_delta_l1": float(np.sum(np.abs(trained.W.data - pre_card_w))),
        "recurrent_conditions_decoder_updated": False,
        "decoder_only_output_trained": True,
        "checkpoint_reproduced_all": all(x["restart_exact"] for x in rows),
        "interpretation": "In-sample source-cue development assay. Human-readable action mappings are provisional; positive scores are NOT independent identity, semantic or out-of-sample evidence.",
        "rows": rows,
    }
