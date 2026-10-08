"""BC01 preview: source-pinned autobiography, lexical cueing, donor recurrence.

Retrieval and cited snippets are external to neural weights. A source hit is
never treated as a semantic entailment verdict. This module is exploratory.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from biocircuit.prototype import Circuit, CircuitConfig
from persona_net.encoding import ACTIONS, ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet

SOURCE_COMMIT = "597fb23473a60eecf2e1b50f79c22bfbea816be5"
SOURCE_BLOB = "718dcc2d5ba4feccdef1690d447edfcebaa9bfb5"
FIXTURE_BLOB = "6e9404d8568b5e3431cdecce0ebd94a73248d6c8"
FIXTURE_PATH = Path(__file__).resolve().parents[1] / "resources/biocircuit/bc01_smoke_v12.jsonl"
SOURCE_PATH = "memories/current/Pretorius_v12_450_Events_Complete.jsonl"
SOURCES = {
    "full": {"count": 450, "blob": SOURCE_BLOB},
    "smoke": {"count": 3, "blob": FIXTURE_BLOB},
}
EVENT_ID = re.compile(r"\bE\d{2}-\d{3}\b", re.IGNORECASE)
TOKENS = re.compile(r"[a-z0-9']+")
STOP = frozenset("a an the and or to of in on at for is are was were with from you your i my me what did do does about who where when why how that this his her he she it they their".split())


def git_blob_sha(raw: bytes) -> str:
    """Verify raw git blob bytes, including final newline and Unicode."""
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


@dataclass(frozen=True)
class Corpus:
    records: tuple[dict, ...]
    source_kind: str
    blob_sha: str

    def by_id(self) -> dict[str, dict]:
        return {item["event_id"]: item for item in self.records}


def load_corpus(path: str | Path = FIXTURE_PATH) -> Corpus:
    raw = Path(path).read_bytes()
    sha = git_blob_sha(raw)
    matches = [name for name, spec in SOURCES.items() if spec["blob"] == sha]
    if len(matches) != 1:
        raise ValueError("Corpus bytes do not match a pinned BC01 source blob")
    kind = matches[0]
    try:
        records = tuple(json.loads(line) for line in raw.decode("utf-8").splitlines())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid UTF-8 JSONL source") from exc
    if len(records) != SOURCES[kind]["count"]:
        raise ValueError("Incorrect source event count")
    required = {"event_id", "episode_id", "chronological_order", "memory_text", "provenance", "recall_cues", "title"}
    ids: set[str] = set()
    orders: list[int] = []
    for item in records:
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError("Missing required source fields")
        eid = item["event_id"]
        if not isinstance(eid, str) or not EVENT_ID.fullmatch(eid) or eid in ids:
            raise ValueError("Duplicate or invalid source event ID")
        if item["provenance"] != "reconstructed":
            raise ValueError("Unexpected provenance class: fail closed")
        if not isinstance(item["memory_text"], str) or not item["memory_text"].strip():
            raise ValueError("Missing first-person narrative")
        if not isinstance(item["recall_cues"], list):
            raise ValueError("Invalid recall cues")
        ids.add(eid)
        orders.append(item["chronological_order"])
    if orders != list(range(1, len(records) + 1)):
        raise ValueError("Source ordering inconsistent")
    return Corpus(records, kind, sha)


def words(s: str) -> set[str]:
    return {t for t in TOKENS.findall(s.lower()) if len(t) > 2 and t not in STOP}


def retrieve(corpus: Corpus, question: str) -> dict:
    """Lexical candidate search, NOT entailment or truth verification."""
    if EVENT_ID.search(question):
        raise ValueError("Event IDs are forbidden in inference queries")
    q = words(question)
    if not q:
        return {"verdict": "unknown", "reason": "no usable lexical cue", "candidate": None}
    ranked: list[tuple[float, dict]] = []
    for event in corpus.records:
        cues = words(" ".join(event["recall_cues"]) + " " + event["title"])
        narrative = words(event["memory_text"])
        overlap = q & (cues | narrative)
        # Require at least two independent informative matches for multiword
        # questions. One shared adjective ("purple") is not source evidence.
        if len(overlap) >= min(2, len(q)) and len(overlap) / len(q) >= 0.5:
            score = (2.0 * len(overlap & cues) + len(overlap & narrative)) / max(1, len(q))
            ranked.append((score, event))
    if not ranked:
        return {"verdict": "unknown", "reason": "no lexical overlap", "candidate": None}
    ranked.sort(key=lambda row: (-row[0], row[1]["chronological_order"]))
    score, event = ranked[0]
    return {
        "verdict": "unknown",
        "reason": "retrieval candidate only; semantic support/refutation unverified",
        "candidate": {
            "event_id": event["event_id"], "provenance": event["provenance"],
            "title": event["title"], "approximate_date": event.get("approximate_date"),
            "source_quote": event["memory_text"][:320], "lexical_score": round(score, 6),
        },
    }


def neural_config(neurons: int, seed: int) -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "config/default.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))["network"]
    if neurons < 128 or neurons % 4:
        raise ValueError("neurons must be >= 128 and divisible by four")
    cfg.update(neurons=neurons, sensory_dim=256, seed=seed)
    return cfg


def make_circuit(neurons: int, seed: int, mode: str) -> Circuit | None:
    if mode == "generic":
        return None
    if mode not in ("local", "global"):
        raise ValueError("Expected local, global or generic")
    active = max(16, (neurons // 32 // 4) * 4)
    return Circuit(CircuitConfig(neurons=neurons, sensory_dim=256, active=active, seed=seed), mode)


def represent(text: str, encoder: ExperienceEncoder, circuit: Circuit | None,
              sensory_override: np.ndarray | None = None) -> np.ndarray:
    """Same lexical source/query interface; optional verified cached source sensory."""
    if sensory_override is None:
        raw = encoder.encode(text).vector
    else:
        if sensory_override.shape != (encoder.sensory_dim,) or not np.all(np.isfinite(sensory_override)):
            raise ValueError("invalid cached sensory vector")
        raw = np.zeros(encoder.input_dim, dtype=np.float32)
        raw[:encoder.sensory_dim] = sensory_override
    if circuit is None:
        return raw
    activity = circuit._activity(raw[:encoder.sensory_dim])
    # Fixed, nonlearned bridge; it cannot serve as an event-ID codebook.
    folded = np.bincount(np.arange(activity.size) % encoder.sensory_dim,
                         weights=activity, minlength=encoder.sensory_dim).astype(np.float32)
    norm = float(np.linalg.norm(folded))
    if norm > 1e-9:
        folded /= norm
    raw[:encoder.sensory_dim] = 0.75 * raw[:encoder.sensory_dim] + 0.25 * folded
    raw[:encoder.sensory_dim] /= max(float(np.linalg.norm(raw[:encoder.sensory_dim])), 1e-9)
    return raw


def settle(net: PlasticRecurrentPersonaNet, stimulus: np.ndarray, ticks: int = 16) -> dict[str, float]:
    net.reset_fast_state()
    for _ in range(ticks):
        net.step(stimulus, reward=0.0, learn=False)
    return net.action_scores()


def report_scores(scores: dict[str, float]) -> dict:
    choice = max(ACTIONS, key=lambda a: scores[a])
    return {"choice": choice, "scores": {k: round(v, 8) for k, v in scores.items()}}


def demo(corpus: Corpus, questions: list[str], neurons: int = 512, seed: int = 1842,
         mode: str = "local", exposures: int = 8, checkpoint: str | Path | None = None,
         shared_sensory: np.ndarray | None = None) -> dict:
    if exposures <= 0:
        raise ValueError("exposures must be positive")
    if any(EVENT_ID.search(q) for q in questions):
        raise ValueError("Event ID leakage in query")
    encoder = ExperienceEncoder(sensory_dim=256)
    if shared_sensory is not None and (
        shared_sensory.shape != (len(corpus.records), 256) or
        shared_sensory.dtype != np.dtype("float32") or
        not np.all(np.isfinite(shared_sensory))
    ):
        raise ValueError("incompatible shared sensory cache")
    cfg = neural_config(neurons, seed)
    circuit = make_circuit(neurons, seed, mode)
    net = PlasticRecurrentPersonaNet(cfg, encoder)
    pristine = net.W.data.copy()
    trained_inputs = [represent(row["memory_text"], encoder, circuit,
                       None if shared_sensory is None else shared_sensory[i])
                      for i, row in enumerate(corpus.records)]
    # Each memory is a first-person reconstructed exposure, not a verified
    # reward-labelled experience. Reward gate is held zero throughout.
    for x in trained_inputs:
        net.reset_fast_state()
        for _ in range(exposures):
            net.step(x, reward=0.0, learn=True)
    delta = net.W.data - pristine
    checkpoint_path = Path(checkpoint) if checkpoint else None
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        net.save(checkpoint_path)
        metadata_path = checkpoint_path.with_suffix(".bc01.json")
        metadata_path.write_text(json.dumps({
            "schema": "BC01-preview-1", "corpus_blob": corpus.blob_sha,
            "source_commit": SOURCE_COMMIT, "mode": mode, "seed": seed,
            "neurons": neurons, "exposures": exposures,
            "encoder": "hashed-lexical-v1; no semantic model",
            "shared_cached_source": bool(shared_sensory is not None),
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved["corpus_blob"] != corpus.blob_sha:
            raise ValueError("Checkpoint corpus mismatch")
        restored = PlasticRecurrentPersonaNet.load(checkpoint_path, cfg, encoder)
    else:
        restored = None
    # The targeted lesion restores only the original recurrent weights:
    # learned bias/homeostasis and the unchanged motor readout remain.
    lesioned = PlasticRecurrentPersonaNet(cfg, encoder)
    lesioned.W.data[:] = pristine
    lesioned.bias[:] = net.bias
    no_plasticity = PlasticRecurrentPersonaNet(cfg, encoder)
    # Matched-tick blank exposure helps expose uniform drift and weight decay.
    # It is not an unrelated-data or independently reviewed control.
    blank_control = PlasticRecurrentPersonaNet(cfg, encoder)
    zero = np.zeros(encoder.input_dim, dtype=np.float32)
    for _ in corpus.records:
        blank_control.reset_fast_state()
        for _ in range(exposures):
            blank_control.step(zero, reward=0.0, learn=True)
    generic = net if mode == "generic" else PlasticRecurrentPersonaNet(cfg, encoder)
    if mode != "generic":
        generic_inputs = [represent(row["memory_text"], encoder, None,
                          None if shared_sensory is None else shared_sensory[i])
                         for i, row in enumerate(corpus.records)]
        for x in generic_inputs:
            generic.reset_fast_state()
            for _ in range(exposures):
                generic.step(x, reward=0.0, learn=True)
    results = []
    for q in questions:
        query_vec = represent(q, encoder, circuit)
        learned = settle(net, query_vec)
        lesion_scores = settle(lesioned, query_vec)
        clean_scores = settle(no_plasticity, query_vec)
        blank_scores = settle(blank_control, query_vec)
        if restored is not None:
            reloaded = settle(restored, query_vec)
            restart_exact = learned == reloaded
        else:
            restart_exact = None
        generic_vec = represent(q, encoder, None)
        control_scores = settle(generic, generic_vec)
        gap = max(abs(learned[a] - lesion_scores[a]) for a in ACTIONS)
        results.append({
            "query": q, "retrieval": retrieve(corpus, q),
            "local_or_selected": report_scores(learned),
            "recurrent_weight_lesion": report_scores(lesion_scores),
            "no_plasticity": report_scores(clean_scores),
            "blank_exposure": report_scores(blank_scores),
            "generic_recurrent": report_scores(control_scores),
            "content_vs_blank_score_max_abs": max(abs(learned[a] - blank_scores[a]) for a in ACTIONS),
            "content_vs_blank_changes_choice": report_scores(learned)["choice"] != report_scores(blank_scores)["choice"],
            "recurrent_delta_score_max_abs": gap,
            "recurrent_delta_changes_choice": report_scores(learned)["choice"] != report_scores(lesion_scores)["choice"],
            "checkpoint_restart_exact": restart_exact,
        })
    return {
        "project": "Pretorius BioCircuit", "version": "BC01-preview-1",
        "status": "exploratory; NO evidence of semantic entailment or identity",
        "corpus_kind": corpus.source_kind, "corpus_events": len(corpus.records),
        "corpus_blob": corpus.blob_sha, "source_repo_commit": SOURCE_COMMIT,
        "encoder": "signed hashed lexical, not a semantic embedding",
        "input_interface": ("validated external cached lexical source and unchanged query encoder"
                            if shared_sensory is not None else
                            "existing lexical encoder; fixed compartment-to-recurrence bridge"),
        "shared_cached_source": bool(shared_sensory is not None),
        "mode": mode, "neurons": neurons, "seed": seed, "exposures_per_memory": exposures,
        "recurrent_changed_synapses": int(np.count_nonzero(delta)),
        "recurrent_delta_l1": float(np.abs(delta).sum()),
        "motor_decoder_trained": False, "reward_signal": "zero; unsupervised exposure",
        "tests": results,
        "limitations": "External lexical index supplies event identity and quote. Action scores have no validated autobiographical policy target. A score change is not a verified behavioral improvement.",
    }
