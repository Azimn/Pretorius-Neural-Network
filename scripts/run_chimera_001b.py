#!/usr/bin/env python3
"""Experiment 001B: explicitly exploratory neural chimera using preserved v0.4 labels.

This is NOT Experiment 001. Uses 100 (not 130) historically authored items.
Original validation and adversarial are previously exposed development diagnostics,
not fresh holdouts. No terminal battery or biography file is opened.

Run with pinned v0.4 checkout:
 python scripts/run_chimera_001b.py --historical-v04 ../historical-v04
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from persona_net.battery import js_divergence, run_items, score_results
from persona_net.encoding import ACTIONS, ExperienceEncoder
from persona_net.network import PlasticRecurrentPersonaNet
from persona_net.phenotype_training import PhenotypeCurriculum
from scripts.run_chimera_001 import (
    CONDITIONS, FRESH_CONDITIONS, atomic_json, fingerprint,
    graft, score, sha256, train_fresh_decoder,
)

EXPLORATORY_CONDITIONS = (*CONDITIONS, *FRESH_CONDITIONS, "shuffled_target_control")


def _items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("items"), list):
        raise ValueError(f"{path}: missing items list")
    return data["items"]


def round_robin(parts: list[list[dict]]) -> list[dict]:
    return [
        part[i] for i in range(max(map(len, parts)))
        for part in parts if i < len(part)
    ]


def load_data(config: dict, historical_root: Path, current_root: Path) -> tuple[dict, dict]:
    """Checksums must match committed v0.4 branch, not old unattainable v1 hash."""
    v04 = historical_root / config["train_source_dir"]
    observed = {}
    for filename, expected in config["train_source_sha256"].items():
        if "/" in filename or "\\" in filename or "terminal" in filename.lower():
            raise ValueError("Forbidden input filename")
        source = v04 / filename
        if not source.is_file():
            raise ValueError(f"Missing pinned original v0.4 source: {source}")
        digest = sha256(source)
        if digest != expected:
            raise ValueError(f"Source integrity mismatch for {source}: {digest} != {expected}")
        observed["v04/" + filename] = digest
    manifest = json.loads((v04 / "manifest.json").read_text(encoding="utf-8"))
    if manifest["train_files"] != config["train_source_files"]:
        raise ValueError("Historical training part order changed")
    if manifest["counts"] != {"train": 100, "validation": 40, "adversarial": 20}:
        raise ValueError("Historical source split sizes changed")
    parts = [_items(v04 / f) for f in config["train_source_files"]]
    if [len(p) for p in parts] != [25] * 4:
        raise ValueError("Expected four original 25-row partitions")
    train = round_robin(parts)
    assert len(train) == config["train_rows"] == 100
    data_root = current_root / "data/phenotype_battery"
    splits = {"train": train}
    for key in ("validation", "adversarial"):
        filename = config[key + "_file"]
        expected = config[key + "_sha256"]
        if "/" in filename or "\\" in filename or "terminal" in filename.lower():
            raise ValueError(f"Forbidden evaluation source filename {filename!r}")
        p = data_root / filename
        if sha256(p) != expected:
            raise ValueError(f"Previously exposed v1 {key} SHA-256 mismatch")
        splits[key] = _items(p)
        observed[key] = expected
    if len(splits["validation"]) != 40 or len(splits["adversarial"]) != 20:
        raise ValueError("Wrong v1 evaluation counts")
    all_rows = splits["train"] + splits["validation"] + splits["adversarial"]
    ids = [row["id"] for row in all_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Train/evaluation identity leakage")
    for row in all_rows:
        if not row.get("scenario", "").strip() or not row.get("target_actions"):
            raise ValueError("Invalid scenario or target")
        PhenotypeCurriculum._normalize_target(row["target_actions"])
    # The historical v0.4 train records do NOT contain a domain field.
    # Domain is a naming convention on IDs, e.g. sovereignty_help_train_01.
    # Validate metadata without changing the source rows or training features.
    domains = []
    for row in train:
        if "_train_" not in row["id"]:
            raise ValueError("Training ID lacks historical *_train_* domain convention")
        domains.append(row["id"].split("_train_", 1)[0])
    if len(set(domains)) != 20 or set(Counter(domains).values()) != {5}:
        raise ValueError("Expected 20 x 5 domain-balanced historical training cards")
    for split, each in (("validation", 2), ("adversarial", 1)):
        domains = Counter(row.get("domain") for row in splits[split])
        if len(domains) != 20 or set(domains.values()) != {each}:
            raise ValueError("Unexpected held-out historical domain counts")
    return splits, observed


def source_priors(train: list[dict], test: list[dict]) -> dict:
    """Static ten-action priors, no encoder, neural substrate or test-label fitting."""
    targets = [PhenotypeCurriculum._normalize_target(row["target_actions"]) for row in train]
    prior = {a: float(np.mean([target[a] for target in targets])) for a in ACTIONS}
    uniform = {a: 1.0 / len(ACTIONS) for a in ACTIONS}
    return {
        "train_action_prior": prior,
        "train_prior_js_similarity": float(np.mean([
            1 - js_divergence(prior, item["target_actions"]) for item in test
        ])),
        "uniform_js_similarity": float(np.mean([
            1 - js_divergence(uniform, item["target_actions"]) for item in test
        ])),
        "note": "Baselines use training-label marginals only. Previously exposed test labels are used for scoring, not fitting.",
    }


def population_readout(net: PlasticRecurrentPersonaNet, encoder: ExperienceEncoder,
                       items: list[dict], config: dict, seed: int, offset: int) -> dict:
    """Decode named action populations without referencing motor_w or motor_b.

    Exploratory direct probe, not a trained or calibrated motor decoder.
    """
    net.rng = np.random.default_rng(seed + config["eval"]["evaluation_seed_offset"] + offset)
    evalcfg = config["eval"]
    zero = np.zeros(encoder.input_dim, dtype=np.float32)
    output = []
    for item in items:
        net.reset_fast_state(noise=evalcfg["noise"])
        for _ in range(evalcfg["settle_ticks"]):
            net.step(zero, reward=0.0, learn=False)
        x = encoder.encode(item["scenario"], scalars=item.get("scalars", {}), action=None).vector
        accum = np.zeros(len(ACTIONS), dtype=np.float64)
        for _ in range(evalcfg["probe_ticks"]):
            net.step(x, reward=0.0, learn=False)
            accum += [float(net.rate[net.action_populations[a]].mean()) for a in ACTIONS]
        accum = np.maximum(accum, 0.0) + 1e-10
        probs = (accum / accum.sum()).tolist()
        output.append({
            "id": item["id"], "domain": item.get("domain"),
            "target_actions": item["target_actions"],
            "action_probabilities": dict(zip(ACTIONS, probs)),
        })
    return score_results(output)


def shuffle_targets(train: list[dict], seed: int) -> list[dict]:
    """Permutation preserves overall target distribution, destroys scenario-label pairing."""
    rng = np.random.default_rng(seed + 702341)
    perm = rng.permutation(len(train))
    if np.array_equal(perm, np.arange(len(train))):
        raise AssertionError("Shuffled control unexpectedly unchanged")
    return [dict(item, target_actions=train[int(perm[i])]["target_actions"])
            for i, item in enumerate(train)]


def train_one_seed(seed: int, config: dict, sets: dict, hashes: dict) -> dict:
    ncfg = dict(config["network"], seed=int(seed))
    encoder = ExperienceEncoder(ncfg["sensory_dim"])
    virgin = PlasticRecurrentPersonaNet(ncfg, encoder)
    mature = copy.deepcopy(virgin)
    trained = PhenotypeCurriculum(sets["train"], encoder).run(
        mature, total_ticks=config["phenotype_training_ticks"], progress_every=0)
    if trained.ticks != config["phenotype_training_ticks"]:
        raise AssertionError("Incorrect actual neural update count")
    outputs = {}
    for name in CONDITIONS:
        r, d = name.split("_")
        net = graft(virgin, mature, r, d)
        outputs[name] = {
            "recurrent_sha256": fingerprint(net, "recurrent"),
            "decoder_sha256": fingerprint(net, "decoder"),
            "validation": score(net, encoder, sets["validation"], config, seed, 0),
            "adversarial": score(net, encoder, sets["adversarial"], config, seed, 1),
        }
    for source, name in (("trained", FRESH_CONDITIONS[0]), ("virgin", FRESH_CONDITIONS[1])):
        net = graft(virgin, mature, source, "virgin")
        fitting = train_fresh_decoder(
            net, encoder, sets["train"], config["fresh_decoder_training_ticks"])
        outputs[name] = {
            "recurrent_sha256": fingerprint(net, "recurrent"),
            "decoder_sha256": fingerprint(net, "decoder"),
            "fresh_decoder_fitting": fitting,
            "validation": score(net, encoder, sets["validation"], config, seed, 0),
            "adversarial": score(net, encoder, sets["adversarial"], config, seed, 1),
        }
    shuffled = shuffle_targets(sets["train"], seed)
    shuffled_mature = copy.deepcopy(virgin)
    shuffled_training = PhenotypeCurriculum(shuffled, encoder).run(
        shuffled_mature, total_ticks=config["phenotype_training_ticks"], progress_every=0)
    net = graft(virgin, shuffled_mature, "trained", "trained")
    outputs["shuffled_target_control"] = {
        "recurrent_sha256": fingerprint(net, "recurrent"),
        "decoder_sha256": fingerprint(net, "decoder"),
        "validation": score(net, encoder, sets["validation"], config, seed, 0),
        "adversarial": score(net, encoder, sets["adversarial"], config, seed, 1),
        "label_permutation_seed": seed + 702341,
        "actual_neural_steps": shuffled_training.ticks,
    }
    # Evaluate the population probe from separate grafts so motor decoder is identical.
    direct = {}
    for r in ("virgin", "trained"):
        net = graft(virgin, mature, r, "virgin")
        direct[r] = {
            "validation": population_readout(net, encoder, sets["validation"], config, seed, 0),
            "adversarial": population_readout(net, encoder, sets["adversarial"], config, seed, 1),
        }
    return {
        "protocol_id": config["protocol_id"],
        "seed": int(seed),
        "source_sha256": hashes,
        "main_neural_training_steps": trained.ticks,
        "phenotype_presentations": trained.presentations,
        "fresh_decoder_neural_steps_each": config["fresh_decoder_training_ticks"],
        "shuffled_control_neural_training_steps": shuffled_training.ticks,
        "trained_recurrent_differs_from_virgin": (
            fingerprint(mature, "recurrent") != fingerprint(virgin, "recurrent")),
        "condition_results": outputs,
        "population_readout_without_motor_decoder": direct,
        "fresh_trained_minus_virgin_validation": (
            outputs[FRESH_CONDITIONS[0]]["validation"]["js_similarity"]
            - outputs[FRESH_CONDITIONS[1]]["validation"]["js_similarity"]),
        "source_status": "exploratory_previously_exposed_v1_evaluation",
        "terminal_battery_touched": False,
        "historical_experiment_001_reproduced": False,
    }


def summarize(per_seed: list[dict], config: dict, priors: dict) -> dict:
    out = {
        "protocol_id": config["protocol_id"],
        "source_status": config["status"],
        "seed_count": len(per_seed),
        "seeds": [r["seed"] for r in per_seed],
        "evaluation_roles": config["eval_roles"],
        "terminal_battery_touched": False,
        "historical_experiment_001_reproduced": False,
        "priors": priors,
        "conditions": {},
        "paired_fresh_trained_minus_virgin_recurrent": [
            float(r["fresh_trained_minus_virgin_validation"]) for r in per_seed
        ],
        "interpretation_limit": (
            "Previously exposed v1 evaluation only. Any apparent decoder advantage "
            "may reflect supervised target fitting on a random recurrent substrate; "
            "paired fresh-decoder controls are mandatory. No fresh independent holdout."
        ),
    }
    for name in EXPLORATORY_CONDITIONS:
        out["conditions"][name] = {}
        for split in ("validation", "adversarial"):
            scores = [r["condition_results"][name][split] for r in per_seed]
            values = [float(x["js_similarity"]) for x in scores]
            out["conditions"][name][split] = {
                "per_seed_js_similarity": values,
                "mean_js_similarity": float(np.mean(values)),
                "mean_domain_macro_js_similarity": float(np.mean([
                    x["domain_macro_js_similarity"] for x in scores
                ])),
                "mean_top1_agreement": float(np.mean([x["top1_agreement"] for x in scores])),
            }
    for cond in ("virgin", "trained"):
        out.setdefault("population_readout", {})[cond] = {
            split: [float(r["population_readout_without_motor_decoder"][cond][split]["js_similarity"])
                    for r in per_seed] for split in ("validation", "adversarial")
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/chimera_001b.json")
    parser.add_argument("--historical-v04", type=Path, required=True,
                        help="Checkout fixed SHA 7b7eb19ad852dc016ee0d370528d903bab78a5cf")
    parser.add_argument("--out", type=Path, default=ROOT / "results/chimera_001b")
    parser.add_argument("--seeds", type=int, nargs="*", help="Explicit subset for smoke only")
    parser.add_argument("--training-ticks", type=int, help="Reduced budget for smoke only")
    parser.add_argument("--probe-ticks", type=int, help="Reduced evaluation for smoke only")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    raw = args.config.read_bytes()
    cfg = json.loads(raw)
    if cfg["protocol_id"] != "chimera-001b-100row-exploratory-pilot-v1":
        raise ValueError("Wrong Experiment 001B protocol")
    if cfg["source_v04_commit"] != "7b7eb19ad852dc016ee0d370528d903bab78a5cf":
        raise ValueError("Unpinned v0.4 source")
    sets, source_hashes = load_data(cfg, args.historical_v04, ROOT)
    if args.preflight_only:
        print("001B preflight OK: 100 train, 40 exposed validation, 20 exposed adversarial")
        return 0
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Empty/duplicate seed list")
    if args.training_ticks is not None:
        if args.training_ticks <= 0 or args.training_ticks % 2:
            raise ValueError("Smoke ticks must be positive and even")
        cfg["phenotype_training_ticks"] = args.training_ticks
        cfg["fresh_decoder_training_ticks"] = args.training_ticks
    if args.probe_ticks is not None:
        if args.probe_ticks <= 0:
            raise ValueError("probe ticks must be positive")
        cfg["eval"]["probe_ticks"] = args.probe_ticks
        cfg["eval"]["settle_ticks"] = args.probe_ticks
    full = (seeds == cfg["seeds"] and args.training_ticks is None
            and args.probe_ticks is None)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "protocol_id": cfg["protocol_id"],
        "configuration_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": source_hashes,
        "run_mode": "full_six_seed_exploratory" if full else "smoke_not_full_evidence",
        "training_order": cfg["train_order"],
        "terminal_battery_touched": False,
        "historical_experiment_001_reproduced": False,
    }
    atomic_json(out / "RUN_METADATA.json", run_meta)
    priors = {k:source_priors(sets["train"], sets[k]) for k in ("validation","adversarial")}
    per_seed = []
    for seed in seeds:
        row = train_one_seed(seed, cfg, sets, source_hashes)
        row["run_mode"] = run_meta["run_mode"]
        row["configuration_sha256"] = run_meta["configuration_sha256"]
        atomic_json(out / f"seed_{seed}.json", row)
        per_seed.append(row)
        print(f"001B {run_meta['run_mode']} seed {seed}: "
              f"intact {row['condition_results']['trained_trained']['validation']['js_similarity']:.5f}; "
              f"fresh paired Δ {row['fresh_trained_minus_virgin_validation']:+.5f}",
              flush=True)
    summary = summarize(per_seed, cfg, priors)
    summary.update(run_meta)
    atomic_json(out / "RUN_REPORT.json", summary)
    print("001B finished; diagnostic nonterminal data only; no Experiment 001 reproduction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
