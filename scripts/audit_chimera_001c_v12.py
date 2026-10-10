#!/usr/bin/env python3
"""Read-only v12 Pretorius memory provenance and annotation queue for 001C.

No neural training, no inferred action targets, no terminal data, and no
episode-disjoint/generalization claim. This is source intake, not an experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_COMMIT = "5364f43dfe3c192b6c13b7bf373405ee7a41b420"
SOURCE_REPO = "Azimn/Pretorius-Connectome"
CORPUS_PATH = "memories/current/Pretorius_v12_450_Events_Complete.jsonl"
SIDECAR_PATH = "memories/annotations/v12_450_sidecars.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def components(events: list[dict]) -> tuple[list[dict], int]:
    parent = {e["event_id"]: e["event_id"] for e in events}
    episodes = {e["event_id"]: e["episode_id"] for e in events}

    def find(s: str) -> str:
        if parent[s] != s:
            parent[s] = find(parent[s])
        return parent[s]

    def union(a: str, b: str) -> None:
        left, right = find(a), find(b)
        if left != right:
            parent[right] = left

    directed_edges = 0
    for e in events:
        for linked in e.get("links_to_prior_events", []):
            if linked not in parent:
                raise ValueError(f"Dangling source memory link {e['event_id']} -> {linked}")
            if linked == e["event_id"]:
                raise ValueError("Self-reference in source memory links")
            union(e["event_id"], linked)
            directed_edges += 1
    groups: dict[str, list[str]] = defaultdict(list)
    for event_id in parent:
        groups[find(event_id)].append(event_id)
    summary = sorted(
        [{"n": len(ids), "episode_count": len({episodes[i] for i in ids}),
          "example_event_id": sorted(ids)[0]} for ids in groups.values()],
        key=lambda x: (-x["n"], x["example_event_id"]),
    )
    return summary, directed_edges


def audit(repo: Path, strict: bool = True) -> tuple[dict, list[dict]]:
    corpus = repo / CORPUS_PATH
    sidecar_file = repo / SIDECAR_PATH
    events = load_jsonl(corpus)
    sidecars = load_jsonl(sidecar_file)

    def unique(rows: list[dict], source: str) -> dict[str, dict]:
        if any(not row.get("event_id") for row in rows):
            raise ValueError(f"{source} contains missing event IDs")
        by_id = {row["event_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"{source} has duplicate event IDs")
        return by_id

    indexed = unique(events, "events")
    annotations = unique(sidecars, "sidecars")
    if set(indexed) != set(annotations):
        raise ValueError("Every event must have exactly one matching sidecar")
    if strict and len(events) != 450:
        raise ValueError(f"V12 archive expected 450 events, got {len(events)}")

    required = ("episode_id", "title", "memory_text", "decisions", "consequences",
                "belief_changes", "relationship_changes", "provenance")
    for row in events:
        if any(key not in row for key in required):
            raise ValueError(f"Core v12 record incomplete: {row['event_id']}")
        if row.get("target_actions"):
            raise ValueError("Source memory unexpectedly has a motor target; requires independent review")
    for row in sidecars:
        if row.get("annotation_status") != "unreviewed_candidate":
            if strict:
                raise ValueError("Annotation review-state changed, audit must be refreshed")

    sizes = Counter(row["episode_id"] for row in events)
    if strict and len(sizes) != 27:
        raise ValueError("Expected exactly 27 v12 autobiographical episodes")
    connected, links = components(events)
    if strict and (len(connected) != 2 or connected[0]["n"] != 448):
        raise ValueError("Unexpected memory graph topology; review before declaring any split")

    queue = []
    for row in sorted(events, key=lambda x: (x["chronological_order"], x["event_id"])):
        annotation = annotations[row["event_id"]]
        queue.append({
            "card_id": "chimera001c:" + row["event_id"],
            "source_repo": SOURCE_REPO,
            "source_commit": SOURCE_COMMIT,
            "source_path": CORPUS_PATH,
            "source_event_id": row["event_id"],
            "episode_id": row["episode_id"],
            "title": row["title"],
            "decision_description": row["decisions"],
            "consequence_description": row["consequences"],
            "belief_change_description": row["belief_changes"],
            "has_causal_inference_status": bool(row.get("causal_inference_status")),
            "links_to_prior_events": row.get("links_to_prior_events", []),
            "cue_candidate_count": len(annotation.get("cue_ids", [])),
            "source_provenance": row.get("provenance"),
            "percept_annotation_status": annotation["annotation_status"],
            "review_status": "needs_independent_behavioral_labels",
            "target_actions": None,
            "reviewer_ids": [],
            "adjudication_status": "not_adjudicated",
            "evaluation_role": "unassigned_memory_graph_entangled",
        })
    report = {
        "audit_id": "chimera-001c-v12-450-source-inventory-v1",
        "status": "unlabeled_candidate_review_queue_only",
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "source_files_sha256": {
            CORPUS_PATH: fingerprint(corpus),
            SIDECAR_PATH: fingerprint(sidecar_file),
        },
        "event_count": len(events),
        "unique_event_ids": len(indexed),
        "episode_count": len(sizes),
        "episode_sizes": dict(sorted(sizes.items())),
        "sidecar_count": len(sidecars),
        "sidecar_review_statuses": dict(sorted(Counter(
            r.get("annotation_status", "missing") for r in sidecars
        ).items())),
        "missing_explicit_causal_inference_status": sum(
            not bool(e.get("causal_inference_status")) for e in events
        ),
        "directed_memory_links": links,
        "connected_components": connected,
        "largest_component_fraction": connected[0]["n"] / len(events),
        "action_label_count": 0,
        "approved_evaluation_item_count": 0,
        "split_policy": (
            "No episode-random or simple episode-held-out split qualifies as a "
            "memory-graph-independent holdout; construct new separately "
            "adjudicated counterfactual behavioral scenarios after freeze."
        ),
        "terminal_battery_touched": False,
        "historical_experiment_001_reproduced": False,
    }
    return report, queue


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-repo", type=Path, required=True,
                   help="Checkout of Pretorius-Connectome at the pinned v12 source commit")
    p.add_argument("--out", type=Path, default=Path("results/chimera_001c"))
    args = p.parse_args()
    report, queue = audit(args.source_repo)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SOURCE_AUDIT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out / "REVIEW_QUEUE.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in queue),
        encoding="utf-8",
    )
    print("001C: events", report["event_count"], "episodes", report["episode_count"],
          "largest_linked_component", report["connected_components"][0]["n"],
          "directed_links", report["directed_memory_links"],
          "unreviewed_sidecars", report["sidecar_count"],
          "missing_causal_status", report["missing_explicit_causal_inference_status"])
    print("No scored action target, no neural training, no independent episode holdout.")


if __name__ == "__main__":
    main()
