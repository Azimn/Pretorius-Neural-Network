#!/usr/bin/env python3
"""Generate answer-free A/B review packets and deliberately invalid review templates.

These are facilitation materials, NOT scored labels, verified independence,
a source-independent test, or model evaluations. Never inject guessed answers.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ENTRY_ROOT=Path(__file__).resolve().parents[1]
if str(_ENTRY_ROOT) not in sys.path:
    sys.path.insert(0,str(_ENTRY_ROOT))

from scripts.validate_chimera_001e_reviews import (
    ROOT, DRAFTS, SOURCE_QUEUE, EXPOSED_VALIDATION, EXPOSED_ADVERSARIAL,
    KNOWN_ACTIONS, load_rows, check_drafts, digest
)

ACTIONS={
    "explore":"Investigate, seek information or test a new possibility",
    "challenge":"Question or resist a claim, practice or request",
    "approach":"Move toward a person, place or interaction",
    "avoid":"Withdraw from an interaction or potential hazard",
    "cooperate":"Coordinate constructively with others",
    "dominate":"Take control or impose a decision on others",
    "create":"Build, design or generate something new",
    "persist":"Continue effort despite obstacles",
    "conceal":"Withhold information or hide an intention",
    "comply":"Follow a rule, instruction or authority demand",
}


def render_packet(drafts: list[dict], queue: list[dict], reviewer: str, seed: int) -> str:
    order=list(drafts)
    random.Random(seed).shuffle(order)
    ref={row["source_event_id"]:row for row in queue}
    if len(order)!=len({d["id"] for d in order}):
        raise ValueError("Duplicate draft identifiers")
    lines=[
        f"# Pretorius 001E, Reviewer {reviewer}: Blind behavioral judgment packet",
        "",
        "These are fictional, assistant-authored counterfactuals inspired by reconstructed ",
        "Pretorius history. No interpretation of his behavior is established fact. ",
        "**Do not look at any neural model predictions, previous experimental results, or",
        "the other reviewer's judgments while completing this packet.**",
        "",
        "For each card, identify plausible action tendencies for the fictional character ",
        "and assign probabilities over the ten abstract actions summing to 1.00. ",
        "Describe ambiguities and alternatives; abstain if the situation cannot",
        "reasonably be mapped. Your own signature and attestation must be recorded",
        "in a separate response file. These packets contain **zero** scored labels.",
        "",
        "## Abstract behavioral action definitions","",
    ]
    for action, meaning in ACTIONS.items():
        lines.append(f"- **{action}**: {meaning}")
    lines += [
        "",
        "A behavioral estimate is not a moral prescription or proof of simulated",
        "consciousness. Scenario text and background snippets were created for this",
        "research; the snippets are not automatically verified autobiographical truth.",
        ""
    ]
    for ordinal,d in enumerate(order,1):
        q=ref[d["source_event_id"]]
        lines += [
            f"## Card {ordinal}: {d['id']}",
            "",
            f"**Scenario SHA-256:** \`{digest(d['scenario'])}\`",
            f"**Source memory reference:** {d['source_event_id']}, "
            f"frozen Pretorius-Connectome v12 \`{d['source_commit']}\`",
            "",
            "### Historical context from the reconstructed archive",
            "",
            f"- Decision recorded: {q['decision_description']}",
            f"- Belief change recorded: {q['belief_change_description']}",
            "",
            "### New hypothetical situation",
            "",
            d["scenario"],
            "",
            "**Independent response:** Choose an action distribution, explain it,",
            "and state uncertainty. Do not consult or predict model output.",
            "",
            "Primary action(s): ____________________",
            "",
            "Probabilities (must sum to 1.00): ____________________",
            "",
            "Evidence / alternative interpretations: ____________________",
            "",
            "---",""
        ]
    return "\n".join(lines)+"\n"


def empty_template(drafts: list[dict], reviewer_role: str) -> str:
    """Intentionally INVALID until filled. Cannot pass adjudication gate."""
    return "".join(json.dumps({
        "candidate_id":d["id"],
        "scenario_sha256":digest(d["scenario"]),
        "reviewer_id":"",
        "attests_independent_read":False,
        "saw_model_predictions":None,
        "target_actions":None,
        "review_note":"",
        "template_role":reviewer_role,
        "template_not_submitted":True,
    },ensure_ascii=False,sort_keys=True)+"\n" for d in drafts)


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"research/chimera_001e/review_packets")
    args=p.parse_args()
    drafts=load_rows(DRAFTS)
    queue=load_rows(SOURCE_QUEUE)
    exposed=[]
    for path in (EXPOSED_VALIDATION,EXPOSED_ADVERSARIAL):
        exposed.extend(json.loads(path.read_text(encoding="utf-8"))["items"])
    check_drafts(drafts,queue,exposed)
    if set(ACTIONS)!=KNOWN_ACTIONS:
        raise ValueError("Action vocabulary diverges from original motor interface")
    args.out.mkdir(parents=True,exist_ok=True)
    for role,seed in (("A",62001),("B",62002)):
        (args.out/f"REVIEWER_{role}.md").write_text(
            render_packet(drafts,queue,role,seed),encoding="utf-8")
        (args.out/f"REVIEWER_{role}.example.jsonl").write_text(
            empty_template(drafts,role),encoding="utf-8")
    print("Built two differently ordered reviewer packets and answer-blank templates.")
    print("Zero approved labels; zero validated ratings; no neural evaluation.")


if __name__=="__main__":
    main()
