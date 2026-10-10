#!/usr/bin/env python3
"""Read-only provenance audit for the missing historical Experiment 001 battery.

Never treats later data as original just because its contents look similar.
The canonical v1 SHA-256 digest is the only acceptable byte-level proof.
No neural model is trained; no terminal data are read.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

V04_COMMIT = "7b7eb19ad852dc016ee0d370528d903bab78a5cf"
PRESERVATION_COMMIT = "a3af4d2433d9df95b2476c556e9e2b9b308944a4"
EXPERIMENT_MAIN_COMMIT = "2f7c698b5a887325cefd52609df746112bdabd6f"
SAFE_SPLITS = ("axis_train", "validation", "adversarial")
TRIAL_ENVELOPES = (
    "items",
    "training_items",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    obj = load(path)
    return obj["items"] if "items" in obj else obj["training_items"]


def round_robin(partitions: list[list[dict]]) -> list[dict]:
    """Invert the preserved four-way stride partitioning without changing rows."""
    return [
        group[index]
        for index in range(max(len(group) for group in partitions))
        for group in partitions
        if index < len(group)
    ]


def check_unique(data: dict[str, list[dict]]) -> dict:
    ids = [item["id"] for group in data.values() for item in group]
    counts = Counter(ids)
    duplicate_ids = [key for key, count in counts.items() if count != 1]
    if duplicate_ids:
        raise ValueError("v0.4 reassembled data contains duplicate IDs: " + str(duplicate_ids[:10]))
    return {"unique_ids": len(counts), "duplicate_ids": duplicate_ids}


def hash_candidates(items: list[dict], expected_sha: str, split: str) -> list[dict]:
    """Try a small *declared* serialization search, not an arbitrary hash rewrite.

    A hit proves exact bytes were regenerated, and may justify restoring that
    source file after independent verification. A miss proves nothing about
    semantic equality or historical row completeness.
    """
    variants = []
    versioned = [
        ("version", "1.0"),
        ("split", "train" if split == "axis_train" else split),
    ]
    # Limit to plausible envelope forms used by the source's JSON loaders.
    for entry in TRIAL_ENVELOPES:
        base = [(entry, items)]
        layouts = [
            base,
            [versioned[0], *base],
            [versioned[0], versioned[1], *base],
            [versioned[1], *base],
            [*base, versioned[0]],
            [*base, versioned[0], versioned[1]],
        ]
        for layout in layouts:
            variants.append(dict(layout))
    matches = []
    for payload in variants:
        for indent in (None, 2, 4):
            for ascii_only in (True, False):
                for sort in (False, True):
                    for compact in (False, True):
                        args = {
                            "ensure_ascii": ascii_only,
                            "sort_keys": sort,
                            "indent": indent,
                        }
                        if compact:
                            args["separators"] = (",", ":") if indent is None else (",", ":")
                        body = json.dumps(payload, **args)
                        for line_ending in ("LF", "CRLF"):
                            text = body if line_ending == "LF" else body.replace("\n", "\r\n")
                            for suffix in ("", "\n", "\r\n"):
                                candidate = (text + suffix).encode("utf-8")
                                if sha256(candidate) == expected_sha:
                                    matches.append({
                                        "envelope_keys": list(payload),
                                        "indent": indent,
                                        "ascii_only": ascii_only,
                                        "sort_keys": sort,
                                        "compact_separators": compact,
                                        "line_ending": line_ending,
                                        "suffix": repr(suffix),
                                        "sha256": expected_sha,
                                    })
    return matches


def serialize_verified_candidate(items: list[dict], match: dict, split: str) -> bytes:
    """Recreate one already SHA-matching JSON representation, no data edits."""
    obj = {}
    for key in match["envelope_keys"]:
        if key == "version":
            obj[key] = "1.0"
        elif key == "split":
            obj[key] = "train" if split == "axis_train" else split
        elif key in ("items", "training_items"):
            obj[key] = items
        else:
            raise ValueError(f"Unsupported historical envelope: {key}")
    args = {
        "ensure_ascii": match["ascii_only"],
        "sort_keys": match["sort_keys"],
        "indent": match["indent"],
    }
    if match["compact_separators"]:
        args["separators"] = (",", ":")
    body = json.dumps(obj, **args)
    if match["line_ending"] == "CRLF":
        body = body.replace("\n", "\r\n")
    return (body + ast.literal_eval(match["suffix"])).encode("utf-8")


def audit(current: Path, v04: Path, preserved: Path, candidates: bool) -> dict:
    spec = load(current / "config/chimera_001.json")
    data_cfg = spec["battery"]
    public = v04 / "data/v0_4"
    public_manifest = load(public / "manifest.json")
    if public_manifest["counts"] != {"train": 100, "validation": 40, "adversarial": 20}:
        raise ValueError("Preserved v0.4 split manifest has unexpected counts")
    data = {}
    order_candidates = {}
    for split, key in [("train", "train_files"), ("validation", "validation_files")]:
        name = "axis_train" if split == "train" else split
        partitions = [rows(public / fn) for fn in public_manifest[key]]
        data[name] = [item for part in partitions for item in part]
        order_candidates[name] = round_robin(partitions)
    data["adversarial"] = rows(public / public_manifest["adversarial_file"])
    order_candidates["adversarial"] = data["adversarial"]
    counts = {name: len(items) for name, items in data.items()}
    if counts != {"axis_train": 100, "validation": 40, "adversarial": 20}:
        raise ValueError("Recovered v0.4 split does not match 100/40/20")
    uniqueness = check_unique(data)
    domain_counts = {}
    for split, expected_each in (("validation", 2), ("adversarial", 1)):
        domains = Counter(item.get("domain") for item in data[split])
        if len(domains) != 20 or set(domains.values()) != {expected_each}:
            raise ValueError(f"{split}: 20-domain stratification violated")
        domain_counts[split] = dict(sorted(domains.items()))
    for row in (item for items in data.values() for item in items):
        if not isinstance(row.get("target_actions"), dict) or sum(
            max(float(v), 0.0) for v in row["target_actions"].values()
        ) <= 0:
            raise ValueError(f"Missing action target for {row.get('id')}")
    original_adv = rows(preserved / "data/phenotype_battery/pretorius_adversarial_v1.json")
    if data["adversarial"] != original_adv:
        raise ValueError("Historical v0.4 adversarial copies differ semantically")
    file_hashes = {
        name: sha256((public / name).read_bytes())
        for name in [
            *public_manifest["train_files"], *public_manifest["validation_files"],
            public_manifest["adversarial_file"], "manifest.json"
        ]
    }
    original_profile = preserved / "data/phenotype_battery/pretorius_profile_v1.json"
    profile_hash = sha256(original_profile.read_bytes())
    if profile_hash != data_cfg["files"]["profile"]["sha256"]:
        raise ValueError("The preservation branch profile does not have the original v1 hash")
    original_adversarial = preserved / "data/phenotype_battery/pretorius_adversarial_v1.json"
    original_adversarial_hash = sha256(original_adversarial.read_bytes())
    if original_adversarial_hash == data_cfg["files"]["adversarial"]["sha256"]:
        raise ValueError("The historical adversarial unexpectedly matches the original; review previous blocker")
    exact = {}
    for split, items in data.items():
        meta = data_cfg["files"][split]
        matches = []
        if candidates:
            for label, sequence in (
                ("concatenated_v04", items),
                ("inverted_strided_partitions", order_candidates[split]),
            ):
                if label == "inverted_strided_partitions" and split == "adversarial":
                    continue
                found = hash_candidates(sequence, meta["sha256"], split)
                for match in found:
                    match["record_order"] = label
                    matches.append(match)
        exact[split] = {
            "required_v1_filename": meta["name"],
            "required_sha256": meta["sha256"],
            "row_count": len(items),
            "candidate_envelope_sha256_matches": matches,
            "claim": (
                "exact v1 SHA-256 source representation found"
                if matches else "candidate semantic source; original v1 hash not yet certified"
            ),
        }
    report = {
        "audit_id": "chimera001-source-recovery-v1",
        "claim": "content-audit-only; NOT an Experiment 001 reproduction",
        "outcome": "candidate-splits-recovered-but-original-v1-battery-unverified",
        "source_pins": {
            "current_runner_main": EXPERIMENT_MAIN_COMMIT,
            "public_v04_payload": V04_COMMIT,
            "preservation_branch": PRESERVATION_COMMIT,
        },
        "terminal_battery_touched": False,
        "original_legacy_train_30_available": False,
        "counts": counts,
        "unique_ids": uniqueness,
        "domains": domain_counts,
        "public_v04_file_sha256": file_hashes,
        "historical_adversarial_file_sha256": original_adversarial_hash,
        "v1_adversarial_expected_sha256": data_cfg["files"]["adversarial"]["sha256"],
        "adversarial_semantically_equal_across_historical_branches": True,
        "v1_profile_sha256_verified": profile_hash,
        "original_v1_exact_hash_recovery": exact,
        "reproduction_gate": "still blocked; 30 legacy training items absent and exact four v1 battery files not authenticated",
        "research_status": "no numerical evaluation was executed",
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--v04", type=Path, required=True)
    parser.add_argument("--preserved", type=Path, required=True)
    parser.add_argument("--candidate-formats", action="store_true")
    parser.add_argument("--restore-verified-adversarial", action="store_true",
                        help="Backward compatible alias for restoring confirmed v1 sources")
    parser.add_argument("--restore-verified-sources", action="store_true",
                        help="Restore ONLY sources with exact precommitted original v1 SHA-256 matches")
    parser.add_argument("--out", type=Path)
    a = parser.parse_args()
    result = audit(a.current, a.v04, a.preserved, a.candidate_formats)
    if a.restore_verified_adversarial or a.restore_verified_sources:
        root = a.v04 / "data/v0_4"
        manifest = load(root / "manifest.json")
        restored = []
        for split, finding in result["original_v1_exact_hash_recovery"].items():
            matches = finding["candidate_envelope_sha256_matches"]
            if not matches:
                continue
            match = matches[0]
            if split == "adversarial":
                items = rows(root / manifest["adversarial_file"])
            else:
                files = manifest["train_files"] if split == "axis_train" else manifest["validation_files"]
                parts = [rows(root / name) for name in files]
                if match["record_order"] == "inverted_strided_partitions":
                    items = round_robin(parts)
                elif match["record_order"] == "concatenated_v04":
                    items = [entry for part in parts for entry in part]
                else:
                    raise ValueError("Unknown source reconstruction order")
            restored_bytes = serialize_verified_candidate(items, match, split)
            original_sha = finding["required_sha256"]
            if sha256(restored_bytes) != original_sha:
                raise ValueError(f"Refusing unauthenticated v1 restoration for {split}")
            canonical = a.current / "data/phenotype_battery" / finding["required_v1_filename"]
            if canonical.exists() and sha256(canonical.read_bytes()) != original_sha:
                raise ValueError(f"Existing noncanonical source {canonical}: review required")
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_bytes(restored_bytes)
            restored.append(split)
            print("AUTHENTICATED_V1_RESTORED", split, canonical, original_sha)
        result["restored_original_v1_splits"] = restored
        result["remaining_unverified_v1_sources"] = [
            result["original_v1_exact_hash_recovery"][split]["required_v1_filename"]
            for split in ("axis_train", "validation", "adversarial") if split not in restored
        ] + ["pretorius_train_v1.json"]
        result["original_v1_status"] = "exact-hash-restoration-partial"
        result["outcome"] = "some original v1 splits restored; 30 legacy training items still missing"
        result["authenticated_original_adversarial_restored"] = "adversarial" in restored
        result["reproduction_gate"] = (
            "blocked: original 130-item complete training battery unavailable; "
            "no neural numerical evaluation has been executed"
        )
    target = a.out or (a.current / "results/chimera_001/SOURCE_RECOVERY_AUDIT.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SOURCE_RECOVERY_AUDIT", target)
    print("COUNTS", json.dumps(result["counts"], sort_keys=True))
    print("ADVERSARIAL_SEMANTIC_PARITY", result["adversarial_semantically_equal_across_historical_branches"])
    print("V1_PROFILE_VERIFIED", bool(result["v1_profile_sha256_verified"]))
    for name, finding in result["original_v1_exact_hash_recovery"].items():
        matches = finding["candidate_envelope_sha256_matches"]
        print("ORIGINAL_V1_CANDIDATE", name, "matched:", len(matches))
        for m in matches[:2]:
            print("MATCH_DETAIL", name, json.dumps(m, sort_keys=True))
    print("EXPERIMENT_001_STATUS: BLOCKED; no neural training performed")


if __name__ == "__main__":
    main()
