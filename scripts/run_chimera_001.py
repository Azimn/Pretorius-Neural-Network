#!/usr/bin/env python3
"""Experiment 001 (v0.3) reconstruction plus a frozen-recurrent fresh-decoder control.

No terminal files are opened. Never change frozen parameters to chase the historical
headline. Phase 2 is gated on Phase 1 reproducing all four validation means.
Run from a checkout: python scripts/run_chimera_001.py
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from persona_net.battery import run_items, score_results
from persona_net.encoding import ACTIONS, SCALAR_KEYS, ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet
from persona_net.phenotype_training import PhenotypeCurriculum

CONDITIONS = (
    "virgin_virgin",
    "trained_trained",
    "trained_virgin",
    "virgin_trained",
)
FRESH_CONDITIONS = (
    "trained_recurrent_fresh_decoder",
    "virgin_recurrent_fresh_decoder",
)


class PreflightError(ValueError):
    """Fail before training if provenance, contents, counts or splits differ."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(net: PlasticRecurrentPersonaNet, part: str) -> str:
    h = hashlib.sha256()
    arrays = (
        (net.W.data, net.W.indices, net.W.indptr, net.Win.data,
         net.Win.indices, net.Win.indptr, net.bias, net.excitatory,
         net.eligibility, *net.action_populations.values())
        if part == "recurrent"
        else (net.motor_w, net.motor_b)
    )
    for x in arrays:
        a = np.asarray(x)
        h.update(str(a.dtype).encode("ascii"))
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def _items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("training_items", payload.get("items"))
    else:
        rows = payload
    if not isinstance(rows, list):
        raise PreflightError(f"{path.name}: missing list of items or training_items")
    return rows


def preflight(config: dict, data_root: Path) -> tuple[dict, dict]:
    spec = config["battery"]
    for key in ("axis_train", "legacy_train", "validation", "adversarial", "profile"):
        name = spec["files"][key]["name"]
        if "/" in name or "\\" in name or "terminal" in name.lower():
            raise PreflightError(f"Forbidden input name: {name!r}")
    sets, hashes = {}, {}
    for key, meta in spec["files"].items():
        path = data_root / meta["name"]
        if not path.is_file():
            raise PreflightError(f"Missing canonical source: {path} (expected SHA-256 {meta['sha256']})")
        observed = sha256(path)
        if observed != meta["sha256"]:
            raise PreflightError(f"Hash mismatch for {path.name}: {observed}, expected {meta['sha256']}")
        hashes[key] = observed
        if key != "profile":
            sets[key] = _items(path)
    profile = json.loads((data_root / spec["files"]["profile"]["name"]).read_text(encoding="utf-8"))
    if len(profile["domains"]) != spec["dimensions"]:
        raise PreflightError("The phenotype profile must specify exactly 20 domains")
    required_counts = {
        "axis_train": spec["axis_train"],
        "legacy_train": spec["legacy_train"],
        "validation": spec["validation"],
        "adversarial": spec["adversarial"],
    }
    for key, expected in required_counts.items():
        if len(sets[key]) != expected:
            raise PreflightError(f"{key}: expected {expected} items, got {len(sets[key])}")
    train = sets["axis_train"] + sets["legacy_train"]
    if len(train) != spec["train"]:
        raise PreflightError("Combined training count differs from pinned 130")
    ids = [item["id"] for key in ("axis_train", "legacy_train", "validation", "adversarial")
           for item in sets[key]]
    if len(ids) != len(set(ids)):
        raise PreflightError("Duplicate item IDs across split boundaries")
    for key in ("validation", "adversarial"):
        domain_counts = {}
        for item in sets[key]:
            domain = item.get("domain")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        expected_per_domain = 2 if key == "validation" else 1
        if len(domain_counts) != spec["dimensions"] or set(domain_counts.values()) != {expected_per_domain}:
            raise PreflightError(f"{key}: expected {expected_per_domain} items in each of 20 domains")
    for item in train + sets["validation"] + sets["adversarial"]:
        if not isinstance(item.get("scenario"), str) or not item["scenario"].strip():
            raise PreflightError(f"{item['id']}: empty scenario")
        try:
            PhenotypeCurriculum._normalize_target(item["target_actions"])
        except (ValueError, KeyError, TypeError) as exc:
            raise PreflightError(f"{item['id']}: invalid target distribution: {exc}") from exc
        if any(k not in SCALAR_KEYS for k in item.get("scalars", {})):
            raise PreflightError(f"{item['id']}: unrecognized environmental scalar")
    return {
        "train": train,
        "validation": sets["validation"],
        "adversarial": sets["adversarial"],
    }, hashes


def graft(founder: PlasticRecurrentPersonaNet,
          mature: PlasticRecurrentPersonaNet,
          recurrent: str, decoder: str) -> PlasticRecurrentPersonaNet:
    """Copy actual constituent arrays, not an adapted decoder or a new topology."""
    if recurrent not in ("virgin", "trained") or decoder not in ("virgin", "trained"):
        raise ValueError("Unknown graft constituent")
    src_r = mature if recurrent == "trained" else founder
    src_d = mature if decoder == "trained" else founder
    out = copy.deepcopy(founder)
    out.W = src_r.W.copy()
    out.Win = src_r.Win.copy()
    out.post_idx = src_r.post_idx.copy()
    out.pre_idx = src_r.pre_idx.copy()
    out.excitatory = src_r.excitatory.copy()
    out.eligibility = src_r.eligibility.copy()
    out.bias = src_r.bias.copy()
    out.action_populations = {k: v.copy() for k, v in src_r.action_populations.items()}
    out.v = src_r.v.copy()
    out.rate = src_r.rate.copy()
    out.tick = src_r.tick
    out.motor_w = src_d.motor_w.copy()
    out.motor_b = src_d.motor_b.copy()
    assert fingerprint(out, "recurrent") == fingerprint(src_r, "recurrent")
    assert fingerprint(out, "decoder") == fingerprint(src_d, "decoder")
    assert not np.shares_memory(out.W.data, src_r.W.data)
    assert not np.shares_memory(out.motor_w, src_d.motor_w)
    return out


def score(net: PlasticRecurrentPersonaNet, encoder: ExperienceEncoder,
          items: list[dict], config: dict, seed: int, offset: int) -> dict:
    settings = config["eval"]
    # The same evaluation noise is used within a seed for each graft.
    net.rng = np.random.default_rng(seed + settings["evaluation_seed_offset"] + offset)
    rows = run_items(net, encoder, items,
                     settle_ticks=settings["settle_ticks"],
                     probe_ticks=settings["probe_ticks"])
    scored = score_results(rows)
    if scored["n"] != len(items):
        raise AssertionError("Not every item was scored")
    return scored


def train_fresh_decoder(net: PlasticRecurrentPersonaNet, encoder: ExperienceEncoder,
                        train: list[dict], ticks: int) -> dict:
    """Replicate PIS exposures and motor supervision with ALL recurrence frozen.

    The same 2-step schedule is used as PhenotypeCurriculum.run, but the second
    step also has learn=False. Only learn_motor_distribution may change weights.
    """
    before_r = fingerprint(net, "recurrent")
    before_d = fingerprint(net, "decoder")
    start_tick = net.tick
    zero_action = slice(encoder.action_offset, encoder.input_dim)
    presentations = 0
    for step in range(0, ticks, 2):
        item = train[presentations % len(train)]
        presentations += 1
        scalars = {k: float(item.get("scalars", {}).get(k, 0.0)) for k in SCALAR_KEYS}
        target = PhenotypeCurriculum._normalize_target(item["target_actions"])
        context = encoder.encode(item["scenario"], scalars=scalars, action=None, outcome=0.0).vector
        x_context = context.copy()
        x_context[zero_action] = 0.0
        net.step(x_context, reward=0.0, learn=False)
        net.learn_motor_distribution(target)
        if step + 1 < ticks:
            teaching = context.copy()
            for ai, action in enumerate(ACTIONS):
                teaching[encoder.action_offset + ai] = target[action]
            net.step(teaching, reward=1.0, learn=False)
    if fingerprint(net, "recurrent") != before_r:
        raise AssertionError("Fresh-decoder fitting modified frozen recurrent substrate")
    if ticks > 0 and fingerprint(net, "decoder") == before_d:
        raise AssertionError("Fresh decoder did not change during fitting")
    return {
        "actual_neural_steps": net.tick - start_tick,
        "presentations": presentations,
        "recurrent_sha256_before_after": before_r,
        "decoder_sha256_initial": before_d,
        "decoder_sha256_final": fingerprint(net, "decoder"),
    }


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def aggregate(per_seed: list[dict], targets: dict, tolerance: float) -> dict:
    dist = {k: [row["phase1"][k]["validation"]["js_similarity"] for row in per_seed]
            for k in CONDITIONS}
    means = {k: float(np.mean(v)) for k, v in dist.items()}
    deviations = {k: means[k] - float(targets[k]) for k in CONDITIONS}
    passes = {k: abs(deviations[k]) <= tolerance for k in CONDITIONS}
    return {
        "seed_distributions": dist,
        "descriptive_means_not_pooled_trials": means,
        "historical_means": targets,
        "delta_vs_historical": deviations,
        "per_condition_pass": passes,
        "all_pass": all(passes.values()),
        "absolute_tolerance": tolerance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/chimera_001.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/phenotype_battery")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/chimera_001")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--phase1-only", action="store_true")
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    out = args.output_dir
    base_report = {
        "protocol_id": config["protocol_id"],
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "data_status": "unverified",
        "terminal_battery_touched": False,
        "claim_status": "not_reproduced",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        splits, source_hashes = preflight(config, args.data_dir)
    except (PreflightError, ValueError, KeyError, json.JSONDecodeError) as error:
        report = dict(base_report, status="blocked_preflight", blocker=str(error))
        atomic_json(out / "STATUS.json", report)
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    base_report.update(data_status="sha256_matched", source_hashes=source_hashes,
                       assumptions=[config["provenance_note"], config["battery"]["train_order"]])
    if args.preflight_only:
        atomic_json(out / "STATUS.json", dict(base_report, status="ready_not_executed"))
        return 0

    per_seed = []
    for seed in config["seeds"]:
        ncfg = dict(config["network"], seed=seed)
        encoder = ExperienceEncoder(ncfg["sensory_dim"])
        founder = PlasticRecurrentPersonaNet(ncfg, encoder)
        mature = copy.deepcopy(founder)
        baseline_r = fingerprint(founder, "recurrent")
        baseline_d = fingerprint(founder, "decoder")
        tr = PhenotypeCurriculum(splits["train"], encoder).run(
            mature, total_ticks=config["phenotype_training_ticks"], progress_every=0)
        if mature.tick != config["phenotype_training_ticks"] or tr.ticks != config["phenotype_training_ticks"]:
            raise AssertionError("Phenotype training step count changed")
        conditions = {}
        for name in CONDITIONS:
            r, d = name.split("_")
            net = graft(founder, mature, r, d)
            conditions[name] = {
                "recurrent_sha256": fingerprint(net, "recurrent"),
                "decoder_sha256": fingerprint(net, "decoder"),
                "validation": score(net, encoder, splits["validation"], config, seed, 0),
                "adversarial": score(net, encoder, splits["adversarial"], config, seed, 1),
            }
        row = {
            "seed": seed,
            "neural_training_steps": tr.ticks,
            "phenotype_presentations": tr.presentations,
            "virgin_recurrent_sha256": baseline_r,
            "virgin_decoder_sha256": baseline_d,
            "trained_recurrent_sha256": fingerprint(mature, "recurrent"),
            "trained_decoder_sha256": fingerprint(mature, "decoder"),
            "phase1": conditions,
            "phase2": None,
        }
        atomic_json(out / f"seed_{seed}.json", dict(base_report, **row))
        per_seed.append(row)
        print(f"seed {seed}: " + ", ".join(
            f"{k}={conditions[k]['validation']['js_similarity']:.5f}" for k in CONDITIONS),
            flush=True)
    gate = aggregate(per_seed, config["narrative_validation_means"],
                     config["reproduction_tolerance_absolute_js_similarity"])
    report = dict(base_report, status="phase1_reproduced" if gate["all_pass"] else "phase1_mismatch",
                  phase1=gate, phase2=None, seeds=config["seeds"])
    atomic_json(out / "RUN_REPORT.json", report)
    if not gate["all_pass"]:
        report["diagnosis"] = (
            "STOP. No Phase 2 and no hyperparameter fitting. Check source manifest, "
            "train ordering, historical hyperparameters and metric definition; "
            "do not overwrite these mismatch results."
        )
        atomic_json(out / "RUN_REPORT.json", report)
        return 3
    if args.phase1_only:
        return 0

    # Important scientific negative control: matched freshly fitted decoder on
    # the original virgin recurrent substrate. Fresh training itself supplies
    # action targets and is sufficient to learn from input features.
    for row in per_seed:
        seed = row["seed"]
        ncfg = dict(config["network"], seed=seed)
        encoder = ExperienceEncoder(ncfg["sensory_dim"])
        founder = PlasticRecurrentPersonaNet(ncfg, encoder)
        mature = copy.deepcopy(founder)
        PhenotypeCurriculum(splits["train"], encoder).run(
            mature, total_ticks=config["phenotype_training_ticks"], progress_every=0)
        phase2 = {}
        for source, name in (("trained", FRESH_CONDITIONS[0]),
                             ("virgin", FRESH_CONDITIONS[1])):
            net = graft(founder, mature, source, "virgin")
            fitting = train_fresh_decoder(net, encoder, splits["train"],
                                          config["phase2"]["decoder_training_ticks"])
            phase2[name] = {
                "fit": fitting,
                "validation": score(net, encoder, splits["validation"], config, seed, 0),
                "adversarial": score(net, encoder, splits["adversarial"], config, seed, 1),
            }
        row["phase2"] = phase2
        atomic_json(out / f"seed_{seed}.json", dict(base_report, **row))
        print(f"fresh seed {seed}: " + ", ".join(
            f"{k}={phase2[k]['validation']['js_similarity']:.5f}" for k in FRESH_CONDITIONS),
            flush=True)
    phase2_dist = {k: [r["phase2"][k]["validation"]["js_similarity"] for r in per_seed]
                   for k in FRESH_CONDITIONS}
    paired = [
        row["phase2"][FRESH_CONDITIONS[0]]["validation"]["js_similarity"]
        - row["phase2"][FRESH_CONDITIONS[1]]["validation"]["js_similarity"]
        for row in per_seed
    ]
    report["status"] = "phase1_reproduced_phase2_complete"
    report["phase2"] = {
        "seed_distributions": phase2_dist,
        "paired_trained_minus_virgin_recurrent": paired,
        "interpretation_rule": (
            "A fresh decoder reaching intact performance does NOT by itself show "
            "recurrent-specific learned information: compare it with the matched "
            "virgin-recurrent fresh-decoder control and report the paired effect."
        ),
    }
    atomic_json(out / "RUN_REPORT.json", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
